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
""" Module for defining benchmark and model enums """

from enum import Enum as pyenum
from typing import Tuple
from sqlalchemy import Column, UniqueConstraint
from sqlalchemy import Enum, Float
from tuna.dbBase.base_class import BASE


#pylint: disable=too-few-public-methods
class FrameworkEnum(pyenum):
  """Represents framework enums"""
  PYTORCH: str = 'Pytorch'
  TENSORFLOW: str = 'Tensorflow'
  MIGRAPH: str = 'MIGraph'
  CAFFE2: str = 'CAFEE2'

  def __str__(self) -> str:
    return self.value


class Framework(BASE):
  """Represents framework table"""
  __tablename__ = "framework"
  __table_args__ = (UniqueConstraint("framework", name="uq_idx"),)
  framework = Column(Enum(FrameworkEnum), nullable=False)
  version = Column(Float, nullable=False)


class ModelEnum(pyenum):
  """Represents model enums"""
  RESNET50: str = 'Resnet50'
  RESNEXT101: str = 'Resnext101'
  VGG16: str = 'Vgg16'
  VGG19: str = 'Vgg19'
  ALEXNET: str = 'Alexnet'
  GOOGLENET: str = 'Googlenet'
  INCEPTION3: str = 'Inception3'
  INCEPTION4: str = 'Inception4'
  MASKRCNN: str = 'Mask-r-cnn'
  SHUFFLENET: str = 'Shufflenet'
  SSD: str = 'ssd'
  MOBILENET: str = 'Mobilenet'
  RESNET101: str = 'Resnet101'
  RESNET152: str = 'Resnet152'
  VGG11: str = 'Vgg11'
  DENSENET: str = 'Densenet'
  DENSENET201: str = 'Densenet201'

  def __str__(self) -> str:
    return self.value


class Model(BASE):
  """Represents model table"""
  __tablename__: str = "model"
  __table_args__: Tuple[UniqueConstraint] = (UniqueConstraint("model",
                                                              "version",
                                                              name="uq_idx"),)
  model = Column(Enum(ModelEnum), nullable=False)
  version = Column(Float, nullable=False)
