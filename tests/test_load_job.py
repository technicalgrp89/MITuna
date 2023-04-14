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
import random

sys.path.append("../tuna")
sys.path.append("tuna")

this_path = os.path.dirname(__file__)

from tuna.miopen.subcmd.load_job import arg_fin_steps, arg_solvers
from tuna.miopen.subcmd.load_job import config_query, compose_query, add_jobs
from tuna.utils.db_utility import get_solver_ids
from tuna.miopen.utils.metadata import ALG_SLV_MAP, TENSOR_PRECISION
from tuna.miopen.db.tables import MIOpenDBTables, ConfigType
from tuna.dbBase.sql_alchemy import DbSession
from tuna.sql import DbCursor
from tuna.utils.logger import setup_logger
from utils import LdJobArgs


#arg_fin_steps function
def test_arg_fin_steps_empty():
  """check that fin_steps attribute remains an empty set when no fin_steps are passed"""
  test_args = argparse.Namespace(fin_steps='')
  arg_fin_steps(test_args)
  assert test_args.fin_steps == set()


def test_arg_fin_steps_none():
  """check that fin_steps attribute remains None when no fin_steps are passed"""
  test_args = argparse.Namespace(fin_steps=None)
  arg_fin_steps(test_args)
  assert test_args.fin_steps == None


def test_arg_fin_steps_tags():
  """check that fin_steps attribute remains None when no fin_steps are passed"""
  test_args = argparse.Namespace(
      fin_steps='miopen_find_compile,miopen_find_eval')
  arg_fin_steps(test_args)
  assert test_args.fin_steps == {'miopen_find_compile', 'miopen_find_eval'}


#arg_solvers function
def test_arg_solvers_none():
  """check that arg_solver attributes when None are passed"""
  args = argparse.Namespace(solvers=None, algo=None)
  logger = logging.getLogger()
  result = arg_solvers(args, logger)
  assert result.solvers == [('', None)]


def test_arg_solvers_slv():
  """check that arg_solver attribute containing solvers are passed"""
  solver_id_map = get_solver_ids()
  random_solver = random.choice(list(get_solver_ids()))
  args = argparse.Namespace(solvers=random_solver, algo=None)
  logger = logging.getLogger()
  result = arg_solvers(args, logger)
  assert result.solvers == [(random_solver, solver_id_map[random_solver])]


def test_arg_solvers_alg():
  """check that arg_solver attribute contains mapped alg:slover passed"""
  solver_id_map = get_solver_ids()
  random_solver = random.choice(list(get_solver_ids()))
  random_algo = random.choice(list(ALG_SLV_MAP.keys()))

  args = argparse.Namespace(solvers=random_solver, algo=random_algo)
  logger = logging.getLogger()
  result = arg_solvers(args, logger)

  assert result.solvers == [(random_solver, solver_id_map[random_solver])]


#config_query/compose functions
def test_cfg():
  """check the config query function for args tags and cmd intake"""
  dbt = MIOpenDBTables(config_type=ConfigType.convolution)

  res = None
  find_conv_job = "SELECT count(*) FROM conv_job;"

  before_cfg_num = 0
  with DbCursor() as cur:
    cur.execute(find_conv_job)
    res = cur.fetchall()
    before_cfg_num = res[0][0]
    print(before_cfg_num)

  args = LdJobArgs
  args.cmd == 'FP32'

  with DbSession() as session:
    cfg_query = config_query(args, session, dbt)
    results = cfg_query.all()

  assert len(results) > 0


#config_compose functions
def test_compose_query():
  """check the config query function for args tags and cmd intake"""
  dbt = MIOpenDBTables(config_type=ConfigType.convolution)

  res = None
  find_conv_job = "SELECT count(*) FROM conv_job;"

  before_cfg_num = 0
  with DbCursor() as cur:
    cur.execute(find_conv_job)
    res = cur.fetchall()
    before_cfg_num = res[0][0]
    print(before_cfg_num)

  args = LdJobArgs
  args.solvers = "GemmFwd1x1_0_1"
  with DbSession() as session:
    cfg_query = config_query(args, session, dbt)
    comp_query = compose_query(args, session, dbt, cfg_query)
    results = comp_query.all()
    print(comp_query.all())

  assert len(results) > 0
