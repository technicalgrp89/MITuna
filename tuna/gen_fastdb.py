"""generate a fastdb dataframe from a finddb dataframe

finddb contains convolution problems and the solvers that solve them.
finddb may list multiple solvers per convolution problem, some solving the
problem faster than the others. fastdb is generated from finddb by looking
at each convolution problem in finddb and keeping only the fastest solver
for it.
"""

import os
import argparse

from tuna.utils import logging
import tuna.utils.tools.df as df_tools
import tuna.utils.tools.io as io_tools
from tuna.gen_finddb import load_finddb
from tuna.utils.db_utility import get_id_solvers
from tuna.utils.ANSI_formatting import ANSIColors
from tuna.utils.finddb_like_utils import get_solver_counts
from tuna.utils.helpers import print_heading, filter_out, map_list, \
  is_substr, invert_dict, sort_dict, as_heading

_DEFAULT_INPUT_DIR = os.path.join(os.getcwd(), 'finddb_')
_DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), 'finddb_')

_, _ID_TO_SOLVER = get_id_solvers()
_SOLVER_TO_ID = invert_dict(_ID_TO_SOLVER)
_DEFAULT_COLS_WITH_CONV_PARAMS = ['fdb_key']


# pylint: disable-next=too-many-locals, too-many-branches
def check_finddb(finddb, cols_with_conv_params=None, strict=False, verbosity=0, tag=''):
  """check finddb and raise erros or log warnings"""
  if cols_with_conv_params is None:
    cols_with_conv_params = _DEFAULT_COLS_WITH_CONV_PARAMS

  # pylint: disable-next=invalid-name
  NO_ENTRIES, INVALID_ENTRIES, OPENCL_ON, MULTIPLE_SESSIONS, DUPLICATE_ENTRIES = 0, 1, 2, 3, 7
  issues = []

  logging.log(f'checking {tag}finddb...', end_char='\r')

  if len(finddb) == 0:
    if strict:
      raise AssertionError(f'{tag}finddb contains no entry')
    logging.error(f'{tag}finddb contains no entry')
    issues.append(NO_ENTRIES)

  if (finddb['valid'] != 1).any():
    if strict:
      raise AssertionError(f'{tag}finddb contains invalid entries')
    logging.warning(f'{tag}finddb contains invalid entries')
    issues.append(INVALID_ENTRIES)

  if finddb['opencl'].any():
    if strict:
      raise AssertionError(f'there are kernels with opencl enabled in {tag}finddb')
    logging.warning(f'there are kernels with opencl enabled in {tag}finddb')
    issues.append(OPENCL_ON)

  if not df_tools.is_col_unique(finddb, col_name='session'):
    if strict:
      raise AssertionError(
          f'data from multiple tuning sessions present in the {tag}finddb')
    logging.warning(
        f'data from multiple tuning sessions present in the {tag}finddb')
    issues.append(MULTIPLE_SESSIONS)

  num_duplicates = finddb.duplicated(subset=cols_with_conv_params +
                                     ['solver']).sum()
  if num_duplicates > 0:
    if verbosity >= 1:
      logging.log(f'{num_duplicates} duplicates found! generating report...', end_char='\r')
      duplicates = finddb[finddb.duplicated(subset=cols_with_conv_params +
                                            ['solver'],
                                            keep=False)]
      duplicates = duplicates.sort_values(cols_with_conv_params +
                                          ['solver', 'kernel_time'])
      duplicates = duplicates.groupby(cols_with_conv_params + ['solver'])

      duplicates_str = as_heading("Duplicates") + '\n'
      for i, (_, df) in enumerate(duplicates): # pylint: disable=invalid-name
        for _, row in df.iterrows():
          conv_params_str = "+ " if i % 2 == 0 else "- "
          for colname, val in zip(cols_with_conv_params,
                                  row[cols_with_conv_params]):
            conv_params_str += f'{colname} {val}, '
          duplicates_str += f"{conv_params_str}\n  solver: {row['solver']},  " +\
                    f"kernel_time: {row['kernel_time']}\n\n"

      logging.log(duplicates_str, silent=True)

    if strict:
      raise AssertionError(
          f'{num_duplicates} duplicate entries detected in {tag}finddb')
    logging.warning(f'{num_duplicates} duplicate entries delected in {tag}finddb')
    issues.append(DUPLICATE_ENTRIES)

  if not issues:
    logging.success(f'{tag}finddb passed all checks!')

  return issues


def describe_fastdb(fastdb, tag=''):
  """ describe fastdb (num. of entries in it, etc.) """
  if len(fastdb) == 0:
    logging.warning(f'{tag}fastdb empty!')
  else:
    logging.info(f'Total entries in {tag}fastdb (=unique configs in {tag}finddb): %d' %
                 len(fastdb))
    logging.info(f'Number of unique solvers in {tag}fastdb: %d' %
                 len(fastdb['solver'].unique()))


def load_fastdb(fastdb_pickle_filename):
  """ load fastdb from pickle """
  fastdb = io_tools.safe_load(None, fastdb_pickle_filename,
                              df_tools.from_pickle)
  describe_fastdb(fastdb)
  return fastdb


def finddb_to_nonthresholded_fastdb(finddb,
                                    cols_with_conv_params=None,
                                    n_fastest=1,
                                    tag=''):
  """ generate fastdb from finddb without thresholding/removing out any solvers """
  if cols_with_conv_params is None:
    cols_with_conv_params = _DEFAULT_COLS_WITH_CONV_PARAMS

  num_duplicates = finddb.duplicated(subset=cols_with_conv_params +
                                     ['solver']).sum()
  if num_duplicates > 0:
    logging.warning(f'resolved the {num_duplicates} duplicate (fdb_key, solver)' +\
            f' pairs found in {tag}finddb by keeping the entries with fastest solver')
  sorted_finddb = finddb.sort_values(cols_with_conv_params + ['kernel_time'])
  return sorted_finddb.groupby(cols_with_conv_params).head(n_fastest)


def is_fundamental_solver(solver):
  """defines fundamental solvers"""

  def is_explicit_gemm(solver):
    return is_substr('Gemm', solver) and not is_substr('ImplicitGemm', solver)

  def is_naive(solver):
    return is_substr('Naive', solver)

  return is_explicit_gemm(solver) or is_naive(solver)


# pylint: disable=too-many-locals, too-many-branches
def gen_fastdb(finddb, threshold=0, keep_fundamental=False, verbosity=0):
  """ generate fastdb with all solvers below threshold removed """
  check_finddb(finddb, verbosity=verbosity)

  logging.log('getting fastest solvers per config...', end_char='\r')
  fastdb = finddb_to_nonthresholded_fastdb(finddb)
  logging.reset_line()

  if threshold > 0:
    finddb_solver_counts = sort_dict(get_solver_counts(finddb))
    fastdb_solver_counts = sort_dict(get_solver_counts(fastdb))

    logging.log('determining solvers below threshold...', end_char='\r')
    solvers_below_threshold = []
    for solver in finddb_solver_counts:
      if solver in fastdb_solver_counts:
        ratio_in_fastdb = fastdb_solver_counts[solver] / sum(
            fastdb_solver_counts.values())
      else:
        ratio_in_fastdb = 0

      if ratio_in_fastdb < (threshold / 100):
        solvers_below_threshold.append(solver)

    logging.reset_line()
    if len(solvers_below_threshold) == 0:
      logging.warning(
          'threshold too loose: none of the solvers are below threshold')
    else:
      if keep_fundamental:
        solvers_to_remove = filter_out(solvers_below_threshold,
                                       is_fundamental_solver)
        logging.log(f'marked {len(solvers_to_remove)} non-fundamental solvers below threshold')
      else:
        solvers_to_remove = solvers_below_threshold
        logging.log(f'marked all of {len(solvers_to_remove)} solvers below threshold')

      solver_ids_to_remove = map_list(solvers_to_remove, _SOLVER_TO_ID)
      thresholded_finddb = finddb[~finddb['solver'].isin(solver_ids_to_remove)]
      logging.log('thresholded-findb generated by removing the marked solvers from finddb')

      thresholded_fastdb = finddb_to_nonthresholded_fastdb(thresholded_finddb, tag='thresholded-')
      logging.success('thresholded-fastdb generated from thresholded-finddb!')

      describe_fastdb(thresholded_fastdb, tag='thresholded-')

      # print summary
      if verbosity >= 1:
        print_heading('SUMMARY', printer=logging.log)
        n = max(len(solver) for solver in finddb_solver_counts) # pylint: disable=invalid-name
        m = len('BELOW-THRESHOLD') # pylint: disable=invalid-name
        for solver in finddb_solver_counts:
          flag_a = 'BELOW-THRESHOLD' if solver in solvers_below_threshold else ''
          flag_b = 'FUNDAMENTAL' if is_fundamental_solver(solver) else ''
          string = f'{solver}'.ljust(n + 3) + f'{flag_a}'.ljust(m + 3) + f'{flag_b}'
          if solver in solvers_to_remove:
            logging.log('- ' + string, formatting=ANSIColors.red)
          else:
            logging.log('+ ' + string)
        logging.log(
            'solver names with a "-" infront got replaced (where a replacement was available)'
        )

      return thresholded_fastdb

  logging.success('fastdb generated!')
  describe_fastdb(fastdb)
  return fastdb


def main():
  """ main """
  default_in_filename = os.path.join(_DEFAULT_INPUT_DIR, 'finddb.pkl')
  default_out_dirname = _DEFAULT_OUTPUT_DIR

  parser = argparse.ArgumentParser(description='Generate fastdb from finddb')
  parser.add_argument(
      '-i',
      '--in',
      type=str,
      default=default_in_filename,
      dest='input',
      help=
      f'filename for the pickled finddb Pandas dataframe (current: {default_in_filename})'
  )
  parser.add_argument(
      '-o',
      '--out',
      type=str,
      default=default_out_dirname,
      dest='output',
      help=
      f'dir to output the pickled fastdb Pandas dataframe to (current: {default_out_dirname})'
  )
  parser.add_argument(
      '-t',
      '--threshold',
      type=float,
      default=0,
      dest='threshold',
      help=
      'replace all solvers w/ lesser frequency than the threshold (default: 0)'
  )
  parser.add_argument(
      '--keep_fundamental',
      action='store_true',
      default=False,
      dest='keep_fundamental',
      help=
      'specify this flag to keep the fundamental solvers despite them being infrequent'
  )
  parser.add_argument('-v',
                      '--verbosity',
                      type=int,
                      default=0,
                      help='higher verbosity => more detailed logs',
                      choices=[0, 1, 2])

  args = parser.parse_args()

  finddb = load_finddb(args.input)
  fastdb = gen_fastdb(finddb, args.threshold, args.keep_fundamental, args.verbosity)

  io_tools.safe_save(fastdb, os.path.join(args.output, 'fastdb.pkl'),
                     df_tools.to_pickle)
  logging.dump_logs(os.path.join(args.output, 'gen_fastdb.log'))


if __name__ == '__main__':
  main()
