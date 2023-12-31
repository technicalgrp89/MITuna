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
"""Module to handle Fin work"""

import json
import os
import tempfile
import functools
import paramiko
from sqlalchemy import func as sqlalchemy_func
from sqlalchemy.exc import IntegrityError, InvalidRequestError  #pylint: disable=wrong-import-order

from tuna.worker_interface import WorkerInterface
from tuna.db_tables import connect_db
from tuna.dbBase.sql_alchemy import DbSession
from tuna.metadata import get_solver_ids, FIN_CACHE
from tuna.metadata import INVERS_DIR_MAP
from tuna.fin_utils import compose_config_obj
from tuna.tables import DBTables
from tuna.config_type import ConfigType
from tuna.utils.db_utility import session_retry


class FinClass(WorkerInterface):
  """Class to provide Tuna support for Fin"""

  # pylint: disable=too-many-instance-attributes

  def __init__(self, **kwargs):
    """Constructor"""
    super().__init__(**kwargs)
    allowed_keys = set([
        'fin_steps', 'local_file', 'fin_outfile', 'fin_infile', 'machine',
        'docker_name', 'version', 'config_type', 'label', 'session_id'
    ])
    self.__dict__.update((key, None) for key in allowed_keys)

    connect_db()
    self.all_configs = []
    self.jobid_to_config = {}
    self.fin_list = []
    self.arch_list = []
    self.supported_fin_steps = ["get_solvers", "applicability"]
    _, self.local_file = tempfile.mkstemp()
    self.fin_infile = self.local_file.split("/tmp/", 1)[1] + ".json"
    _, self.local_output = tempfile.mkstemp()
    self.fin_outfile = self.local_output.split("/tmp/", 1)[1] + ".json"
    self.fin_steps = []
    self.machine = None
    self.docker_name = None
    self.cnx = None
    self.config_type = ConfigType.convolution if self.config_type is None else self.config_type
    self.label = None
    self.session_id = None

    self.multiproc = False

    self.__dict__.update(
        (key, value) for key, value in kwargs.items() if key in allowed_keys)

    self.dbt = DBTables(session_id=self.session_id,
                        config_type=self.config_type)

  def chk_abort_file(self):
    """Checking presence of abort file to terminate processes immediately"""
    abort_reason = []
    if os.path.exists(f'/tmp/miopen_abort_{self.machine.arch}'):
      abort_reason.append(self.machine.arch)

    if os.path.exists(f'/tmp/miopen_abort_mid_{self.machine.id}'):
      abort_reason.append('mid_' + str(self.machine.id))
    if abort_reason:
      for reason in abort_reason:
        self.logger.warning('/tmp/mipen_abort_%s file found, returning', reason)
      return True

    return False

  def compose_fincmd(self):
    """Helper function to compose fin docker cmd"""
    if self.machine.local_machine:
      # Skip the copy and use the /tmp/* version of the files
      fin_ifile = self.local_file
      fin_ofile = self.local_output
    else:
      fin_ifile = FIN_CACHE + "/" + self.fin_infile
      #Currently not used, but will be used in the future
      #machine_cfg = get_qts_machine_data(self.machine.id, self.machine.hostname,
      #                                   self.logger)
      try:
        self.logger.info("Fin: copying local fin input_file: %s to remote %s",
                         self.local_file, fin_ifile)
        # TODO: remove redundant file copies  # pylint: disable=fixme
        self.cnx.ssh.open_sftp().put(self.local_file, fin_ifile)
        self.logger.info("Fin: Successfully copied to remote")
      except paramiko.ssh_exception.SSHException:
        self.logger.warning('unable to connect to remote %s', fin_ifile)
      except IOError:
        self.logger.warning('unable to receive file: %s skipping ... ',
                            fin_ifile)

      fin_ofile = FIN_CACHE + "/" + self.fin_outfile
    bash_cmd = f"/opt/rocm/bin/fin -i {fin_ifile} -o {fin_ofile}"
    self.logger.info('Executing fin cmd: %s', bash_cmd)
    return bash_cmd

  def get_solvers(self):
    """Getting solvers from MIOpen to update Tuna DB"""
    self.fin_steps = ['get_solvers']
    solvers = self.get_fin_results()
    if solvers is None:
      return False
    if 'all_solvers' not in solvers[1]:
      self.logger.error('all_solvers key not found in fin output')
    self.parse_solvers(solvers[1]['all_solvers'])

    return True

  def get_fin_results(self):
    """Helper function to launch fin docker cmd, cat output of the cmd and parse the json"""
    # pylint: disable=broad-except
    result = None

    if self.prep_fin_input(self.local_file, to_file=True):
      fin_cmd = self.compose_fincmd()
      ret_code, out, err = self.exec_docker_cmd(fin_cmd)
      if ret_code > 0:
        self.logger.warning('Err executing cmd: %s', fin_cmd)
        self.logger.warning(out)
        raise Exception(
            f'Failed to execute fin cmd: {fin_cmd} err: {err.read()}')

      result = self.parse_out()

    return result

  def parse_out(self):
    """Parse fin output helper function"""
    # pylint: disable=broad-except
    result = None
    if not self.machine.local_machine:
      fin_outfile = FIN_CACHE + "/" + self.fin_outfile
      # TODO: This should be copied back out using cat is bad # pylint: disable=fixme
      _, ssh_stdout, _ = self.exec_command(f"cat {fin_outfile}")
      result_json = []

      for line in ssh_stdout:
        result_json.append(line)
      try:
        result = json.loads('\n'.join(result_json))
      except Exception as err:
        self.logger.warning('Err loading fin json: %s', err)
        return None
    else:
      with open(self.local_output) as out_file:  # pylint: disable=unspecified-encoding
        try:
          result = json.load(out_file)
        except Exception as err:
          self.logger.error('Unable to load fin json file %s', err)
          for line in out_file:
            self.logger.error(line)
          return None

    return result

  def applicability(self):
    """Getting applicability from MIOpen to update Tuna DB"""
    self.fin_steps = ['applicability']
    applic_res = self.get_fin_results()
    if applic_res is None:
      return False

    self.parse_applicability(applic_res)

    return True

  def set_all_configs(self, idx=0, num_blk=1):
    """Gathering all configs from Tuna DB to set up fin input file"""
    with DbSession() as session:
      query = session.query(
          self.dbt.config_table).filter(self.dbt.config_table.valid == 1)

      if self.label:
        query = query.filter(self.dbt.config_table.id == self.dbt.config_tags_table.config)\
            .filter(self.dbt.config_tags_table.tag == self.label)

      #order by id for splitting configs into blocks
      query = query.order_by(self.dbt.config_table.id)
      rows = query.all()

      block_size = len(rows) // num_blk  #size of the config block
      extra = len(rows) % num_blk  #leftover configs, don't divide evenly
      start = idx * block_size  #start of a process block
      end = (idx + 1) * block_size
      #distributing leftover configs to processes
      if idx < extra:
        start += idx
        end += 1 + idx
      else:
        start += extra
        end += extra

      self.logger.info("cfg workdiv: proc %s, start %s, end %s", self.gpu_id,
                       start, end)

      if start >= len(rows):
        return False

      #copy the configs for this process
      if end >= len(rows):
        subarr = rows[start:]
      else:
        subarr = rows[start:end]

      for row in subarr:
        r_dict = compose_config_obj(row, self.config_type)
        if self.config_type == ConfigType.batch_norm:
          r_dict['direction'] = row.get_direction()
        self.all_configs.append(r_dict)

    if self.all_configs is None:
      return False

    return True

  def create_dumplist(self):
    """Creating json dump to be used as fin input file"""
    self.fin_list = []
    if len(self.fin_steps) == 1 and self.fin_steps == ["get_solvers"]:
      self.fin_list = [{"steps": self.fin_steps}]
      self.logger.info("Creating dumplist for: %s", self.fin_steps[0])
      return True

    if "applicability" in self.fin_steps:
      self.logger.info("Creating dumplist for: %s", self.fin_steps[0])
      idx = 0
      num_blk = 1
      if self.multiproc:
        idx = self.gpu_id
        num_blk = self.num_procs.value

      if not self.set_all_configs(idx, num_blk):
        return False
      return self.compose_fin_list()

    self.logger.error("Fin steps not recognized: %s", self.fin_steps)
    self.logger.info("Fin steps recognized are: %s", self.supported_fin_steps)
    return False

  def compose_fin_list(self):
    """Helper function to set fin_list for dump"""

    arch = self.dbt.session.arch
    ncu = self.dbt.session.num_cu
    for cfg in self.all_configs:
      self.fin_list.append({
          "steps": self.fin_steps,
          "arch": arch,
          "num_cu": ncu,
          "config_tuna_id": cfg["id"],
          "config": cfg,
          "direction": int(INVERS_DIR_MAP[cfg["direction"]])
      })
    return True

  def dump_json(self, outfile, to_file=True):
    """Dumping json to outfile"""
    if to_file is True:
      if not os.path.exists(outfile):
        os.mknod(outfile)
      with open(outfile, 'w') as fout:  # pylint: disable=unspecified-encoding
        fout.write("[\n")
        i = 0
        while i < len(self.fin_list):
          json_out = json.dumps(self.fin_list[i])
          fout.write(json_out)
          if i != len(self.fin_list) - 1:
            fout.write(',\n')
          i += 1

        fout.write("\n]")
      self.logger.info('Fin input file written to %s', outfile)
    else:
      jdump = json.dumps(self.fin_list)
      return jdump

    return True

  def prep_fin_input(self, outfile=None, to_file=True):
    """Main function in Fin that produces Fin input file"""

    self.cnx = self.machine.connect(self.chk_abort_file())
    ret = False
    if outfile is None:
      outfile = "fin_input.json"
    if self.create_dumplist():
      ret = self.dump_json(outfile, to_file)
    else:
      self.logger.warning("Could not create dumplist for Fin input file")

    return ret

  def insert_applicability(self, session, json_in):
    """write applicability to sql"""
    _, solver_id_map_h = get_solver_ids()
    for elem in json_in:
      if "applicable_solvers" in elem.keys():
        #remove old applicability
        # pylint: disable=comparison-with-callable
        app_query = session.query(self.dbt.solver_app)\
          .filter(self.dbt.solver_app.session == self.session_id)\
          .filter(self.dbt.solver_app.config == elem["input"]["config_tuna_id"])
        # pylint: enable=comparison-with-callable
        app_query.update({self.dbt.solver_app.applicable: 0},
                         synchronize_session='fetch')
        if not elem["applicable_solvers"]:
          self.logger.warning("No applicable solvers for %s",
                              elem["input"]["config_tuna_id"])
        for solver in elem["applicable_solvers"]:
          try:
            solver_id = solver_id_map_h[solver]
            obj = app_query.filter(
                self.dbt.solver_app.solver == solver_id).first()  # pylint: disable=W0143
            if obj:
              obj.applicable = 1
            else:
              new_entry = self.dbt.solver_app(
                  solver=solver_id,
                  config=elem["input"]["config_tuna_id"],
                  session=self.session_id,
                  applicable=1)
              session.add(new_entry)
          except KeyError:
            self.logger.warning('Solver %s not found in solver table', solver)
            self.logger.info("Please run 'go_fish.py --update_solver' first")
            return False

    self.logger.info('Commit bulk transaction, please wait')
    session.commit()
    return True

  def parse_applicability(self, json_in):
    """Function to parse fin outputfile and populate DB with results"""
    self.logger.info('Parsing fin solver applicability output...')
    if json_in is None:
      self.logger.error("JSON file returned from Fin is empty")
      return False

    #break down commits into smaller packets
    pack_i = 0
    pack = []
    all_packs = []
    for elem in json_in:
      pack.append(elem)
      pack_i += 1
      if pack_i == 100:
        all_packs.append(pack)
        pack = []
        pack_i = 0
    if pack:
      all_packs.append(pack)

    with DbSession() as session:

      def actuator(func, pack):
        return func(session, pack)

      for pack in all_packs:
        session_retry(session, self.insert_applicability,
                      functools.partial(actuator, pack=pack), self.logger)

    with DbSession() as session:
      query = session.query(sqlalchemy_func.count(self.dbt.solver_app.id))
      query = query.filter(self.dbt.solver_app.session == self.session_id)  # pylint: disable=W0143
      sapp_count = query.one()[0]
      self.logger.warning(
          "Finished parsing solver applicability, new session size: %d entries",
          sapp_count)
    return True

  def invalidate_solvers(self, sids, max_id):
    """Helper function to invalidate solver in DB that are not present in Fin outputfile"""
    solver_ids_invalid = []
    with DbSession() as session:
      i = 1
      try:
        while i <= max_id:
          #if solver has been removed
          if i not in sids:
            solver_ids_invalid.append(i)
            session.query(self.dbt.solver_table).filter(
                self.dbt.solver_table.id == i).update(
                    {self.dbt.solver_table.valid: 0})
            session.commit()
          i += 1
      except IntegrityError as err:
        self.logger.warning("DB err occurred %s", err)

    return solver_ids_invalid

  def add_new_solvers(self, solvers):
    """Add new solvers to db and return the key for latest solver"""

    max_id = 1
    sids = []
    with DbSession() as session:
      for slv_map in solvers:
        idx = int(slv_map['id'])
        solver = slv_map['name']
        tunable = int(slv_map['tunable'])
        config_type = slv_map['type']
        try:
          sids.append(idx)
          if idx > max_id:
            max_id = idx

          new_s = self.dbt.solver_table(id=idx,
                                        solver=solver,
                                        valid=1,
                                        tunable=tunable,
                                        config_type=config_type,
                                        is_dynamic=slv_map['dynamic'])
          session.add(new_s)
          session.commit()
        except IntegrityError:
          self.logger.info(
              "Duplicate entry, updating solver %s: valid=1, tunable=%s",
              solver, tunable)
          session.rollback()
          session.query(self.dbt.solver_table).filter(
              self.dbt.solver_table.id == idx).update({
                  self.dbt.solver_table.valid: 1,
                  self.dbt.solver_table.solver: solver,
                  self.dbt.solver_table.tunable: tunable
              })
          session.commit()
        except InvalidRequestError as err2:
          self.logger.info("DB err occurred: %s", err2)

    return max_id, sids

  def parse_solvers(self, solvers):
    """Function to parse solvers from fin output file"""
    # TODO: Refactor such that we query all the solvers # pylint: disable=fixme
    # from the db once then only insert/invalidate the new/invalid one
    max_id, sids = self.add_new_solvers(solvers)

    solver_ids_invalid = []
    if len(sids) != max_id:
      solver_ids_invalid = self.invalidate_solvers(sids, max_id)
      self.logger.info("invalid solvers: %s", solver_ids_invalid)

    s_count = 0
    with DbSession() as session:
      query = session.query(sqlalchemy_func.count(self.dbt.solver_table.id))
      s_count = query.one()[0]

    if max_id != s_count:
      #Note: we canot update invalid solvers missing from DB bc MIOpen does not report these
      self.logger.info(
          "Solver table missing some invalid solvers, please check MIOpens solver.cpp \
          file for solvers that have been invalidated and are missing from your DB"
      )
      self.logger.info("Current invalid solvers: %s", solver_ids_invalid)

    return True

  def step(self):
    """Inner loop for Process run defined in worker_interface"""
    self.multiproc = True
    if "applicability" in self.fin_steps:
      self.applicability()

    self.multiproc = False
    return False
