###############################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

import os
import sys
import argparse
import logging
import pytest
import itertools

from sqlalchemy.orm import Query

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)

from tuna.miopen.subcmd.export_db import (
    get_filename, get_base_query, get_fdb_query, get_pdb_query,
    get_fdb_alg_lists, build_miopen_fdb, write_fdb, export_fdb,
    build_miopen_kdb, insert_perf_db_sqlite, create_sqlite_tables, get_cfg_dict)
from tuna.utils.db_utility import DB_Type
from tuna.miopen.db.tables import MIOpenDBTables, ConfigType
from tuna.dbBase.sql_alchemy import DbSession
from tuna.utils.db_utility import get_id_solvers
from utils import add_test_session, DummyArgs, CfgEntry, TensorEntry, build_fdb_entry


session_id = add_test_session()
fdb_entry = build_fdb_entry(session_id)
args = DummyArgs()
args.session_id = session_id
args.config_type = ConfigType.convolution
args.golden_v = None
args.arch = "gfx900"
args.num_cu = 64
args.opencl = True
args.config_tag = None
args.filename = None
dbt = MIOpenDBTables(session_id=args.session_id, config_type=args.config_type)
logger = logging.getLogger("test_logger")


def test_get_file():
  db_type = DB_Type.FIND_DB

  expeted_filename = "tuna_1.0.0/gfx900_64.OpenCL.fdb.txt"
  actual_filename = get_filename(args.arch, args.num_cu, args.filename,
                                 args.opencl, db_type)

  assert actual_filename == expeted_filename, f"expected {expeted_filename}, but got {actual_filename}"


def test_get_base_query():
  query = get_base_query(dbt, args, logger)
  assert query is not None, "Query object is None"
  assert isinstance(
      query,
      Query), f"epected query to be an instance of Query, Got {type(query)}"


def test_get_fdb_query():
  fdb_query = get_fdb_query(dbt, args, logger)
  assert fdb_query is not None, "Query object is None"
  assert isinstance(
      fdb_query,
      Query), f"epected query to be an instance of Query, Got {type(fdb_query)}"


def test_get_fdb_alg_lists():
  fdb_query = get_fdb_query(dbt, args, logger)
  alg_lists = get_fdb_alg_lists(fdb_query, logger)
  assert alg_lists is not None, f"expected a retrived an fdb alg list, Got {type(test_get_fdb_alg_lists)}"


def test_get_pdb_query():
  pdb_query = get_pdb_query(dbt, args, logger)
  assert pdb_query is not None, "Query object is None"
  assert isinstance(
      pdb_query,
      Query), f"epected query to be an instance of Query, Got {type(pdb_query)}"


def test_build_export_miopen_fdp():
  fdb_query = get_fdb_query(dbt, args, logger)
  alg_lists = get_fdb_alg_lists(fdb_query, logger)
  miopen_fdb = build_miopen_fdb(alg_lists, logger)
  fdb_file = write_fdb(args.arch, args.num_cu, args.opencl, miopen_fdb,
                       args.filename)
  fdb_exported = export_fdb(dbt, args, logger)
  assert miopen_fdb is not None, f"failed to build miopen_fdb, Got {type(miopen_fdb)}"
  assert fdb_file is not None
  assert fdb_exported is not None


def test_buid_miopen_kdb():
  fdb_query = get_fdb_query(dbt, args, logger)
  alg_lists = get_fdb_alg_lists(fdb_query, logger)
  build_mioopen_kdp = build_miopen_kdb(dbt, alg_lists, logger)
  assert build_mioopen_kdp is not None


def test_create_sqlite_tables():
  cnx, local_path = create_sqlite_tables(args.arch, args.num_cu, args.filename)
  cur = cnx.cursor()
  cur.execute(
      "SELECT name from sqlite_master WHERE type='table' AND (name='config' or name='perf_db')"
  )
  table_names = cur.fetchall()
  cur.close()
  assert len(table_names) == 2
  assert ('config') in table_names[0]
  assert ('perf_db') in table_names[1]

  cur = cnx.cursor()
  cur.execute(
      "SELECT name from sqlite_master WHERE type='index' AND (name='idx_config' OR name='idx_perf_db')"
  )
  index_names = cur.fetchall()
  cur.close()
  assert len(index_names) == 2
  assert ('idx_config') in index_names[0]
  assert ('idx_perf_db') in index_names[1]
  cnx.close()
  os.remove(local_path)


def test_insert_perf_db_sqlite():
  ins_cfg_id = 1
  perf_db_entry = build_fdb_entry(session_id)
  cnx, local_path = create_sqlite_tables(args.arch, args.num_cu, args.filename)
  perf_db_dict = insert_perf_db_sqlite(cnx, perf_db_entry, ins_cfg_id)
  _, ID_SOLVER_MAP = get_id_solvers()
  expected_data = {
      "config": ins_cfg_id,
      "solver": ID_SOLVER_MAP[perf_db_entry.solver],
      "params": perf_db_entry.params
  }

  for key, value in expected_data.items():
    assert perf_db_dict[key] == value

  cur = cnx.cursor()
  cur.execute("SELECT config, solver FROM perf_db WHERE config=?",
              (ins_cfg_id,))
  inserted_data = cur.fetchone()
  cur.close()
  os.remove(local_path)
  assert inserted_data is not None, "No data was inserted into perf_db table"
  assert inserted_data == tuple(
      itertools.islice(expected_data.values(), 2)
  ), f"expected inserted data {tuple(itertools.islice(expected_data.values(),2))}, but got {inserted_data}"

def test_get_cfg_dict():

  cfg_entry = CfgEntry()
  tensor_entry = TensorEntry()

  cfg_dict = get_cfg_dict(cfg_entry, tensor_entry)

  assert isinstance(cfg_dict, dict)
  assert 'tensor_id_1' in cfg_dict
  assert 'tensor_id_2' in cfg_dict
  assert cfg_dict['tensor_id_1'] == 'cfg_value_1'
  assert cfg_dict['tensor_id_2'] == 'cfg_value_2'


def test_export_db():
  session_id = add_test_session(arch='gfx90a',
                                num_cu=110,
                                label='pytest_export_db')

  args = DummyArgs()
  args.session_id = session_id
  args.config_type = ConfigType.convolution
  args.golden_v = None
  args.opencl = False
  args.config_tag = None
  args.filename = 'outfile'

  dbt = MIOpenDBTables(session_id=args.session_id)

  if args.session_id:
    args.arch = dbt.session.arch
    args.num_cu = dbt.session.num_cu

  with DbSession() as session:
    fdb_entry = dbt.find_db_table()
    fdb_entry.solver = 1
    fdb_entry.config = 1
    fdb_entry.opencl = False
    fdb_entry.session = session_id
    fdb_entry.fdb_key = 'key1'
    fdb_entry.params = 'params'
    fdb_entry.kernel_time = 10
    fdb_entry.workspace_sz = 0
    fdb_entry.kernel_group = 11359
    session.add(fdb_entry)

    fdb_entry2 = dbt.find_db_table()
    fdb_entry2.solver = 1
    fdb_entry2.config = 2
    fdb_entry2.opencl = False
    fdb_entry2.session = session_id
    fdb_entry2.fdb_key = 'key2'
    fdb_entry2.params = 'params'
    fdb_entry2.kernel_time = 20
    fdb_entry2.workspace_sz = 0
    fdb_entry2.kernel_group = 11360
    session.add(fdb_entry2)

    kcache = dbt.kernel_cache()
    kcache.kernel_name = 'kname1'
    kcache.kernel_args = 'arg1'
    kcache.kernel_blob = bytes('blob', 'utf-8')
    kcache.kernel_hash = 0
    kcache.uncompressed_size = 0
    kcache.kernel_group = 11359
    session.add(kcache)

    kcache2 = dbt.kernel_cache()
    kcache2.kernel_name = 'kname2'
    kcache2.kernel_args = 'arg2'
    kcache2.kernel_blob = bytes('blob', 'utf-8')
    kcache2.kernel_hash = 0
    kcache2.uncompressed_size = 0
    kcache2.kernel_group = 11360
    session.add(kcache2)

    session.commit()

  query = get_fdb_query(dbt, args)
  miopen_fdb = build_miopen_fdb(query)
  miopen_kdb = build_miopen_kdb(dbt, miopen_fdb)

  for key, solvers in sorted(miopen_fdb.items(), key=lambda kv: kv[0]):
    if key == 'key1':
      assert solvers[0].config == 1
    elif key == 'key2':
      assert solvers[0].config == 2
    else:
      assert False

  export_fdb(dbt, args)

  output_fp = open(
      get_filename(args.arch, args.num_cu, args.filename, args.opencl,
                   DB_Type.FIND_DB))
  for line in output_fp:
    key, vals = line.split('=')
    assert key in ('key1', 'key2')
    if key == 'key1':
      assert float(vals.split(':')[1].split(',')[0]) == 10.0
    elif key == 'key2':
      assert float(vals.split(':')[1].split(',')[0]) == 20.0

  for kern in miopen_kdb:
    if kern.kernel_group not in (11359, 11360):
      if kern.kernel_group == 11359:
        assert kern.kernel_name == 'kname1'
      elif kern.kernel_group == 11360:
        assert kern.kernel_name == 'kname2'

  pdb_entries = get_pdb_query(dbt, args).all()
  for entry, _ in pdb_entries:
    if entry.fdb_key == 'key1':
      assert entry.config == 1
    elif entry.fdb_key == 'key2':
      assert entry.config == 2
    else:
      assert False
