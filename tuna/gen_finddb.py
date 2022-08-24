"""pull finddb down into a pandas dataframe object from the tuna database"""

import os
import argparse
from enum import Enum

import pandas as pd
from sqlalchemy import and_

from tuna.utils import logging
from tuna.tables import DBTables
import tuna.utils.tools.io as io_tools
import tuna.utils.tools.df as df_tools
from tuna.config_type import ConfigType
from tuna.utils.helpers import pretty_list
from tuna.dbBase.sql_alchemy import DbSession


_DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), 'finddb_')


def load_finddb(finddb_pickle_filename, tag=''):
  """ loads finddb dataframe from a pickle

  Attributes:
    finddb_pickle_filename: pickle of finddb dataframe
    tag: tag for finddb (just for better logging)
  """
  finddb = io_tools.safe_load(None, finddb_pickle_filename,
                              df_tools.from_pickle)
  describe_finddb(finddb, tag)
  return finddb


def describe_finddb(finddb, tag=''):
  """ logs a description of finddb

  Attributes:
    finddb: finddb dataframe
    tag: tag for finddb (just for better logging)
  """
  if len(finddb) == 0:
    logging.warning('finddb empty!')
  else:
    logging.info(f'{tag}finddb corresponds to session IDs: %s' %
                 pretty_list(finddb['session'].unique()))
    logging.info(f'Total entries in {tag}finddb: %d' % len(finddb))
    logging.info(f'Total unique solvers in {tag}finddb: %d' %
                 len(finddb['solver'].unique()))


def gen_finddb(session_ids, invalid_too, opencl_only):
  """pulls down finddb into a pandas dataframe from the mysql database

  Attributes:
    session_ids: finddb dataframe will only contain entries from these session_ids
    invalid_too: finddb dataframe will contain invalid entries too
    opencl_only: finddb dataframe will only contain entries with opencl
  """
  db_tables = DBTables(config_type=ConfigType.convolution)
  finddb = db_tables.find_db_table

  logging.info(f'tuna database name: {os.environ["TUNA_DB_NAME"]}')
  logging.info(f'finddb table name: {finddb.__tablename__}')

  with DbSession() as session:
    logging.log(f'quering {finddb.__tablename__}...', end_char='\r')
    query = session.query(finddb)
    query = query.filter(
        and_(finddb.kernel_time != -1, finddb.workspace_sz != -1))
    if session_ids is not None:
      for session_id in session_ids:
        # pylint: disable-next=comparison-with-callable
        query = query.filter(finddb.session == session_id)
    if invalid_too is False:
      query = query.filter(finddb.valid == True) # pylint: disable=singleton-comparison
    if opencl_only is True:
      query = query.filter(finddb.opencl == True) # pylint: disable=singleton-comparison

    query = query.order_by(finddb.update_ts.desc(), finddb.config)

    logging.reset_line()
    logging.success('query processed!')

    logging.log('reading query into a dataframe...', end_char='\r')
    df = pd.read_sql(query.statement, session.bind) # pylint: disable=invalid-name
    logging.reset_line()

    describe_finddb(df)

    return df


class FinddbParsing:
  """utilities to add and parse finddb-related Arguments"""

  SIGNATURE = '__hasFinddbArgs__'

  class ARGNAMES(Enum):
    """defines finddb-related Arguments"""
    SESSION_IDS = 'session_ids'
    INVALID_TOO = 'invalid_too'
    OPENCL_ONLY = 'opencl_only'

  @classmethod
  def set_finddb_args(cls, parser):
    """adds finddb-related Arguments to a parser

    Attributes:
      parser: argparse.ArgumentParser object
    """
    # SESSION_IDS
    parser.add_argument(
        f'--{FinddbParsing.ARGNAMES.SESSION_IDS.value}',
        type=int,
        nargs='*',
        default=None,
        dest=FinddbParsing.ARGNAMES.SESSION_IDS.value,
        help=
        'IDs of tuning sessions to fetch finddb for (default: all tuning sessions)'
    )
    # INVALID_TOO
    parser.add_argument(f'--{FinddbParsing.ARGNAMES.INVALID_TOO.value}',
                        action='store_true',
                        default=False,
                        dest=FinddbParsing.ARGNAMES.INVALID_TOO.value,
                        help='dump both valid and invalid kernels (default: False)')
    # OPENCL_ONLY
    parser.add_argument(f'--{FinddbParsing.ARGNAMES.OPENCL_ONLY.value}',
                        action='store_true',
                        default=False,
                        dest=FinddbParsing.ARGNAMES.OPENCL_ONLY.value,
                        help='only dump kernels that use opencl extension (default: False)')

    # sign the parser
    setattr(parser, FinddbParsing.SIGNATURE, True)

  @classmethod
  def get_finddb_args(cls, args):
    """parses Finddb-related Arguments from a parser

    Attributes:
      parser: argparse.ArgumentParser object
    """

    return {
        argname.value: getattr(args, argname.value)
        for argname in FinddbParsing.ARGNAMES
    }


def main():
  """ main """
  default_out_dirname = _DEFAULT_OUTPUT_DIR

  parser = argparse.ArgumentParser(
      description='Fetches finddb and exports it as a Pandas DataFrame')
  parser.add_argument(
      '-o',
      '--out',
      type=str,
      default=default_out_dirname,
      help=
      f'directory for the output pickled Pandas Dataframe (current: {default_out_dirname})'
  )
  FinddbParsing.set_finddb_args(parser)

  args = parser.parse_args()

  finddb = gen_finddb(**FinddbParsing.get_finddb_args(args))

  io_tools.safe_save(finddb, os.path.join(args.out, 'finddb.pkl'),
                     df_tools.to_pickle)
  logging.dump_logs(os.path.join(args.out, 'gen_finddb.log'))


if __name__ == '__main__':
  main()
