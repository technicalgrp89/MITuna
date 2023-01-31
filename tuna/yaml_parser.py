#!/usr/bin/env python3
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
"""! @brief YAML custom parser for yaml files to support multiple tuning steps
     per yaml file. """

import os
import yaml
from tuna.miopen.yaml_parser import parse_miopen_yaml
from tuna.example.yaml_parser import parse_example_yaml
from tuna.libraries import Library


def parse_yaml(filename, lib):
  """Parses input yaml file and returns 1 or multiple yaml files(when library support
     for multiple yaml files is provided).
     Multiple yaml files are returned when more that 1 step per initial yaml file
     are specified."""
  yaml_dict = None
  yaml_files = []
  with open(os.path.expanduser(filename)) as stream:
    try:
      yaml_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
      print(exc)

  #if Library not specified here, no custom parsing function will be used
  if lib == Library.MIOPEN:
    yaml_files = parse_miopen_yaml(yaml_dict)
  elif lib == Library.EXAMPLE:
    yaml_files = parse_example_yaml(yaml_dict)
  else:
    #return current yaml file without custom parsing
    return filename

  return yaml_files
