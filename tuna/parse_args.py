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
""" Module to centralize command line argument parsing """
import argparse
from enum import Enum
from typing import List
from tuna.helper import ConfigType


class TunaArgs(Enum):
  """ Enumeration of all the common argument supported by setup_arg_parser """
  ARCH = 0
  NUM_CU = 1
  DIRECTION = 2
  VERSION = 3
  CONFIG_TYPE = 4


def setup_arg_parser(desc: str, arg_list: List[TunaArgs]):
  """ function to aggregate common command line args """
  parser = argparse.ArgumentParser(description=desc)
  if TunaArgs.ARCH in arg_list:
    parser.add_argument(
        '-a',
        '--arch',
        type=str,
        dest='arch',
        default=None,
        required=False,
        help='Architecture of machines',
        choices=['gfx900', 'gfx906', 'gfx908', 'gfx1030', 'gfx90a'])
  if TunaArgs.NUM_CU in arg_list:
    parser.add_argument('-n',
                        '--num_cu',
                        dest='num_cu',
                        type=int,
                        default=None,
                        required=False,
                        help='Number of CUs on GPU',
                        choices=[36, 56, 60, 64, 104, 110, 120])
  if TunaArgs.DIRECTION in arg_list:
    parser.add_argument(
        '-d',
        '--direction',
        type=str,
        dest='direction',
        default=None,
        help='Direction of tunning, None means all (fwd, bwd, wrw), \
                          fwd = 1, bwd = 2, wrw = 4.',
        choices=[None, '1', '2', '4'])
  if TunaArgs.CONFIG_TYPE in arg_list:
    parser.add_argument('-C',
                        '--config_type',
                        dest='config_type',
                        help='Specify configuration type',
                        default=ConfigType.convolution,
                        choices=ConfigType,
                        type=ConfigType)
  return parser


def parse_args_export_db():
  """Function to parse arguments"""
  parser = setup_arg_parser('Convert MYSQL find_db to text find_dbs' \
    'architecture', [TunaArgs.ARCH, TunaArgs.NUM_CU, TunaArgs.VERSION])
  parser.add_argument('-c',
                      '--opencl',
                      dest='opencl',
                      action='store_true',
                      help='Use OpenCL extension',
                      default=False)
  parser.add_argument(
      '--session_id',
      action='store',
      type=int,
      dest='session_id',
      help=
      'Session ID to be used as tuning tracker. Allows to correlate DB results to tuning sessions'
  )
  parser.add_argument('--config_tag',
                      dest='config_tag',
                      type=str,
                      help='import configs based on config tag',
                      default=None)
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

  return args


def parse_args_populate_golden():
  """! Function to parse arguments"""
  parser = setup_arg_parser('Populate golden table based on session_id',
                            [TunaArgs.CONFIG_TYPE])
  parser.add_argument(
      '--session_id',
      action='store',
      type=int,
      dest='session_id',
      help=
      'Session ID to be used as tuning tracker. Allows to correlate DB results to tuning sessions'
  )
  parser.add_argument('--golden_v',
                      action='store',
                      type=int,
                      dest='golden_v',
                      help='Golden miopen version')
  args = parser.parse_args()
  return args
