#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################
"""Module to export find_db to txt file"""
import sqlite3
import os
from collections import OrderedDict
import base64

from tuna.dbBase.sql_alchemy import DbSession
from tuna.tables import DBTables
from tuna.metadata import SQLITE_PERF_DB_COLS
from tuna.utils.db_utility import get_id_solvers, DB_Type
from tuna.utils.utility import arch2targetid
from tuna.utils.logger import setup_logger
from tuna.parse_args import TunaArgs, setup_arg_parser
from tuna.analyze_parse_db import get_config_sqlite, insert_solver_sqlite, mysql_to_sqlite_cfg
from tuna.fin_utils import compose_config_obj

DIR_NAME = {'F': 'Fwd', 'B': 'BwdData', 'W': 'BwdWeights'}

# Setup logging
LOGGER = setup_logger('export_db')

_, ID_SOLVER_MAP = get_id_solvers()


def parse_args():
  """Function to parse arguments"""
  parser = setup_arg_parser('Convert MYSQL find_db to text find_dbs' \
    'architecture', [TunaArgs.ARCH, TunaArgs.NUM_CU, TunaArgs.VERSION])

  group_ver = parser.add_mutually_exclusive_group(required=True)
  group_ver.add_argument(
      '--session_id',
      dest='session_id',
      type=int,
      help=
      'Session ID to be used as tuning tracker. Allows to correlate DB results to tuning sessions'
  )
  group_ver.add_argument(
      '--golden_v',
      dest='golden_v',
      type=int,
      help='export from the golden table using this version number')

  parser.add_argument('--config_tag',
                      dest='config_tag',
                      type=str,
                      help='import configs based on config tag',
                      default=None)
  parser.add_argument('-c',
                      '--opencl',
                      dest='opencl',
                      action='store_true',
                      help='Use OpenCL extension',
                      default=False)
  parser.add_argument('--filename',
                      dest='filename',
                      help='Custom filename for DB dump',
                      default=None)

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-k',
                     '--kern_db',
                     dest='kern_db',
                     action='store_true',
                     help='Serialize Kernel Database',
                     default=False)
  group.add_argument('-f',
                     '--find_db',
                     dest='find_db',
                     action='store_true',
                     help='Serialize Find Database',
                     default=False)
  group.add_argument('-p',
                     '--perf_db',
                     dest='perf_db',
                     action='store_true',
                     help='Serialize Perf Database',
                     default=False)
  args = parser.parse_args()

  if args.golden_v and not (args.arch and args.num_cu):
    parser.error('arch and num_cu must be set with golden_v')

  return args


def get_filename(arch, num_cu, filename, ocl, db_type):
  """Helper function to compose filename"""
  version = "1.0.0"
  tuna_dir = f'tuna_{version}'
  if not os.path.exists(tuna_dir):
    os.makedirs(tuna_dir)
  final_name = f"{tuna_dir}/{arch}_{num_cu}"
  if num_cu > 64:
    final_name = f'{tuna_dir}/{arch}{num_cu:x}'
  if filename:
    final_name = f'{tuna_dir}/{filename}'

  if db_type == DB_Type.FIND_DB:
    # pylint: disable-next=consider-using-f-string ; more readable
    extension = '.{}.fdb.txt'.format('OpenCL' if ocl else 'HIP')
  elif db_type == DB_Type.KERN_DB:
    extension = '.kdb'
  else:
    extension = ".db"

  final_name = f"{final_name}{extension}"

  return final_name


def get_base_query(dbt, args):
  """ general query for fdb/pdb results """
  src_table = dbt.find_db_table
  if args.golden_v is not None:
    src_table = dbt.golden_table

  with DbSession() as session:
    query = session.query(src_table, dbt.config_table)
    if args.golden_v is not None:
      query = query.filter(src_table.golden_miopen_v == args.golden_v)\
              .filter(src_table.arch == args.arch)\
              .filter(src_table.num_cu == args.num_cu)
      LOGGER.info("golden_miopen_v: %s, arch: %s, num_cu: %s", args.golden_v,
                  args.arch, args.num_cu)
    else:
      query = query.filter(src_table.session == dbt.session.id)
      LOGGER.info("rocm_v : %s", dbt.session.rocm_v)
      LOGGER.info("miopen_v : %s", dbt.session.miopen_v)

    query = query.filter(src_table.valid == 1)\
        .filter(src_table.opencl == args.opencl)\
        .filter(src_table.config == dbt.config_table.id)\
        .filter(src_table.solver == dbt.solver_table.id)

    if args.config_tag:
      LOGGER.info("config_tag : %s", args.config_tag)
      query = query.filter(dbt.config_tags_table.tag == args.config_tag)\
          .filter(dbt.config_table.config == dbt.config_table.id)

    LOGGER.info("base db query returned: %s", len(query.all()))

  return query


def get_fdb_query(dbt, args):
  """ Helper function to create find db query
  """
  src_table = dbt.find_db_table
  if args.golden_v is not None:
    src_table = dbt.golden_table

  query = get_base_query(dbt, args)
  query = query.filter(src_table.kernel_time != -1)\
      .filter(src_table.workspace_sz != -1)

  query = query.order_by(src_table.fdb_key, src_table.update_ts.desc())

  return query


def get_pdb_query(dbt, args):
  """Compose query to get perf_db rows based on filters from args"""
  src_table = dbt.find_db_table
  if args.golden_v is not None:
    src_table = dbt.golden_table

  query = get_base_query(dbt, args)
  query = query.filter(dbt.solver_table.tunable == 1)\
      .filter(src_table.params != '')

  LOGGER.info("pdb query returned: %s", len(query.all()))

  return query


def get_fdb_alg_lists(query):
  """return dict with key: fdb_key + alg_lib, val: solver list"""
  find_db = OrderedDict()
  solvers = {}
  for fdb_entry, _ in query.all():
    fdb_key = fdb_entry.fdb_key
    if fdb_key not in solvers:
      solvers[fdb_key] = {}
    if fdb_entry.solver in solvers[fdb_key].keys():
      LOGGER.warning("Skipped duplicate solver: %s : %s with ts %s vs prev %s",
                     fdb_key, fdb_entry.solver, fdb_entry.update_ts,
                     solvers[fdb_key][fdb_entry.solver])
      continue
    solvers[fdb_key][fdb_entry.solver] = fdb_entry.update_ts

    fdb_key_alg = (fdb_entry.fdb_key, fdb_entry.alg_lib)
    lst = find_db.get(fdb_key_alg)
    if not lst:
      find_db[fdb_key_alg] = [fdb_entry]
    else:
      lst.append(fdb_entry)

  return find_db


def build_miopen_fdb(fdb_alg_lists):
  """ create miopen find db object for export
  """
  total_entries = len(fdb_alg_lists)
  num_fdb_entries = 0
  miopen_fdb = OrderedDict()
  for fdbkey_alg, alg_entries in fdb_alg_lists.items():
    fdb_key = fdbkey_alg[0]
    num_fdb_entries += 1
    #pick fastest solver for each algorithm
    alg_entries.sort(key=lambda x: float(x.kernel_time))
    fastest_entry = alg_entries[0]
    lst = miopen_fdb.get(fdb_key)
    if not lst:
      miopen_fdb[fdb_key] = [fastest_entry]
    else:
      lst.append(fastest_entry)

    if num_fdb_entries % (total_entries // 10) == 0:
      LOGGER.info("FDB count: %s, fdb: %s, cfg: %s, slv: %s", num_fdb_entries,
                  fastest_entry.fdb_key, fastest_entry.config,
                  ID_SOLVER_MAP[fastest_entry.solver])

  LOGGER.warning("Total number of entries in Find DB: %s", num_fdb_entries)

  return miopen_fdb


def write_fdb(arch, num_cu, ocl, find_db, filename=None):
  """
  Serialize find_db map to plain text file in MIOpen format
  """
  file_name = get_filename(arch, num_cu, filename, ocl, DB_Type.FIND_DB)

  with open(file_name, 'w') as out:  # pylint: disable=unspecified-encoding
    for key, solvers in sorted(find_db.items(), key=lambda kv: kv[0]):
      solvers.sort(key=lambda x: float(x.kernel_time))
      lst = []
      # for alg_lib, solver_id, kernel_time, workspace_sz in solvers:
      for rec in solvers:
        # pylint: disable-next=consider-using-f-string ; more reable
        lst.append('{alg}:{},{},{},{alg},{}'.format(ID_SOLVER_MAP[rec.solver],
                                                    rec.kernel_time,
                                                    rec.workspace_sz,
                                                    'not used',
                                                    alg=rec.alg_lib))
      out.write(f"{key}={';'.join(lst)}\n")
  return file_name


def export_fdb(dbt, args):
  """Function to export find_db to txt file
  """
  query = get_fdb_query(dbt, args)
  fdb_alg_lists = get_fdb_alg_lists(query)
  miopen_fdb = build_miopen_fdb(fdb_alg_lists)

  return write_fdb(args.arch, args.num_cu, args.opencl, miopen_fdb,
                   args.filename)


def build_miopen_kdb(dbt, find_db):
  """ create miopen kernel db object for export
  """
  num_fdb_entries = 0
  num_kdb_blobs = 0
  kern_db = []
  with DbSession() as session:
    total = len(find_db.items())
    last_pcnt = 0
    for _, entries in find_db.items():
      num_fdb_entries += 1
      entries.sort(key=lambda x: float(x.kernel_time))
      fastest_slv = entries[0]
      query = session.query(dbt.kernel_cache)\
          .filter(dbt.kernel_cache.kernel_group == fastest_slv.kernel_group)
      for kinder in query.all():
        num_kdb_blobs += 1
        kern_db.append(kinder)
      pcnt = int(num_fdb_entries * 100 / total)
      if pcnt > last_pcnt:
        LOGGER.warning("Building db: %s%%, blobs: %s", pcnt, num_kdb_blobs)
        last_pcnt = pcnt

  LOGGER.warning("Total FDB entries: %s, Total blobs: %s", num_fdb_entries,
                 num_kdb_blobs)
  return kern_db


def write_kdb(arch, num_cu, kern_db, filename=None):
  """
  Write blob map to sqlite
  """
  file_name = get_filename(arch, num_cu, filename, None, DB_Type.KERN_DB)
  if os.path.isfile(file_name):
    os.remove(file_name)

  conn = sqlite3.connect(file_name)
  cur = conn.cursor()
  cur.execute(
      "CREATE TABLE `kern_db` (`id` INTEGER PRIMARY KEY ASC,`kernel_name` TEXT NOT NULL,"
      "`kernel_args` TEXT NOT NULL,`kernel_blob` BLOB NOT NULL,`kernel_hash` TEXT NOT NULL,"
      "`uncompressed_size` INT NOT NULL);")
  cur.execute(
      "CREATE UNIQUE INDEX `idx_kern_db` ON kern_db(kernel_name, kernel_args);")

  ins_list = []
  arch_ext = arch2targetid(arch)
  for kern in kern_db:
    name = kern.kernel_name
    args = kern.kernel_args
    #check if extensions should be added
    if not name.endswith('.o'):
      name += ".o"
    if not "-mcpu=" in args:
      if not name.endswith('.mlir.o'):
        args += f" -mcpu={arch_ext}"

    ins_key = (name, args)
    if ins_key not in ins_list:
      ins_list.append(ins_key)
      cur.execute(
          "INSERT INTO kern_db (kernel_name, kernel_args, kernel_blob, kernel_hash, "
          "uncompressed_size) VALUES(?, ?, ?, ?, ?);",
          (name, args, base64.b64decode(
              kern.kernel_blob), kern.kernel_hash, kern.uncompressed_size))

  conn.commit()
  cur.close()
  conn.close()

  LOGGER.warning("Inserted blobs: %s", len(ins_list))
  return file_name


def export_kdb(dbt, args):
  """
  Function to export the kernel cache
  """
  query = get_fdb_query(dbt, args)
  fdb_alg_lists = get_fdb_alg_lists(query)
  miopen_fdb = build_miopen_fdb(fdb_alg_lists)

  LOGGER.info("Building kdb.")
  kern_db = build_miopen_kdb(dbt, miopen_fdb)

  LOGGER.info("write kdb to file.")
  return write_kdb(args.arch, args.num_cu, kern_db, args.filename)


def create_sqlite_tables(arch, num_cu, filename=None):
  """create sqlite3 tables"""
  local_path = get_filename(arch, num_cu, filename, None, DB_Type.PERF_DB)

  cnx = sqlite3.connect(local_path)

  cur = cnx.cursor()
  cur.execute(
      "CREATE TABLE IF NOT EXISTS `config` (`id` INTEGER PRIMARY KEY ASC,`layout` TEXT NOT NULL,"
      "`data_type` TEXT NOT NULL,`direction` TEXT NOT NULL,`spatial_dim` INT NOT NULL,"
      "`in_channels` INT NOT NULL,`in_h` INT NOT NULL,`in_w` INT NOT NULL,`in_d` INT NOT NULL,"
      "`fil_h` INT NOT NULL,`fil_w` INT NOT NULL,`fil_d` INT NOT NULL,"
      "`out_channels` INT NOT NULL, `batchsize` INT NOT NULL,"
      "`pad_h` INT NOT NULL,`pad_w` INT NOT NULL,`pad_d` INT NOT NULL,"
      "`conv_stride_h` INT NOT NULL,`conv_stride_w` INT NOT NULL,`conv_stride_d` INT NOT NULL,"
      "`dilation_h` INT NOT NULL,`dilation_w` INT NOT NULL,`dilation_d` INT NOT NULL,"
      "`bias` INT NOT NULL,`group_count` INT NOT NULL)")
  cur.execute(
      "CREATE TABLE IF NOT EXISTS `perf_db` (`id` INTEGER PRIMARY KEY ASC,`solver` TEXT NOT NULL,"
      "`config` INTEGER NOT NULL, `params` TEXT NOT NULL)")

  cur.execute(
      "CREATE UNIQUE INDEX IF NOT EXISTS `idx_config` ON config( layout,data_type,direction,"
      "spatial_dim,in_channels,in_h,in_w,in_d,fil_h,fil_w,fil_d,out_channels,"
      "batchsize,pad_h,pad_w,pad_d,conv_stride_h,conv_stride_w,conv_stride_d,"
      "dilation_h,dilation_w,dilation_d,bias,group_count )")
  cur.execute(
      "CREATE UNIQUE INDEX IF NOT EXISTS `idx_perf_db` ON perf_db(solver, config)"
  )

  cur.close()
  cnx.commit()
  return cnx, local_path


def get_cfg_dict(cfg_entry, tensor_entry):
  """compose config_dict"""
  cfg_dict = compose_config_obj(cfg_entry)

  if cfg_entry.valid == 1:
    cfg_dict = mysql_to_sqlite_cfg(cfg_dict)

  ext_dict = tensor_entry.to_dict(ommit_valid=True)
  ext_dict.pop('id')
  cfg_dict.update(ext_dict)

  #bias is always 0
  cfg_dict['bias'] = 0

  return dict(cfg_dict)


def insert_perf_db_sqlite(cnx, perf_db_entry, ins_cfg_id):
  """insert perf_db entry into sqlite"""
  perf_db_dict = perf_db_entry.to_dict()
  perf_db_dict['config'] = ins_cfg_id
  perf_db_dict = {
      k: v for k, v in perf_db_dict.items() if k in SQLITE_PERF_DB_COLS
  }
  perf_db_dict['solver'] = ID_SOLVER_MAP[perf_db_dict['solver']]

  insert_solver_sqlite(cnx, perf_db_dict)

  return perf_db_dict


def export_pdb(dbt, args):
  """ export perf db from mysql to sqlite """
  cnx, local_path = create_sqlite_tables(args.arch, args.num_cu, args.filename)
  num_perf = 0
  query = get_pdb_query(dbt, args)
  cfg_map = {}
  db_entries = query.all()
  total_entries = len(db_entries)
  for perf_db_entry, cfg_entry in db_entries:
    if cfg_entry.id in cfg_map:
      ins_cfg_id = cfg_map[cfg_entry.id]
    else:
      cfg_dict = get_cfg_dict(cfg_entry, cfg_entry.input_t)
      #filters cfg_dict by SQLITE_CONFIG_COLS, inserts cfg if missing
      ins_cfg_id = get_config_sqlite(cnx, cfg_dict)
      cfg_map[cfg_entry.id] = ins_cfg_id

    pdb_dict = insert_perf_db_sqlite(cnx, perf_db_entry, ins_cfg_id)
    num_perf += 1

    if num_perf % (total_entries // 10) == 0:
      cnx.commit()
      LOGGER.info("PDB count: %s, mysql cfg: %s, pdb: %s", num_perf,
                  cfg_entry.id, pdb_dict)

  cnx.commit()
  LOGGER.warning("Total number of entries in Perf DB: %s", num_perf)

  return local_path


def main():
  """Main module function"""
  args = parse_args()
  result_file = ''
  dbt = DBTables(session_id=args.session_id)

  if args.session_id:
    args.arch = dbt.session.arch
    args.num_cu = dbt.session.num_cu

  if args.find_db:
    result_file = export_fdb(dbt, args)
  elif args.kern_db:
    result_file = export_kdb(dbt, args)
  elif args.perf_db:
    result_file = export_pdb(dbt, args)

  print(result_file)


if __name__ == '__main__':
  main()
