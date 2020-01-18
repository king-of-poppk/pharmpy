import re
from io import StringIO
from pathlib import Path

import pandas as pd

import pharmpy.math


class NONMEMTableFile:
    '''A NONMEM table file that can contain multiple tables
    '''
    def __init__(self, path):
        path = Path(path)
        suffix = path.suffix
        self.tables = []
        with open(str(path), 'r') as tablefile:
            current = []
            for line in tablefile:
                if line.startswith("TABLE NO."):
                    if current:
                        self._add_table(current, suffix)
                    current = [line]
                else:
                    current.append(line)
            self._add_table(current, suffix)
        self._count = 0

    def __iter__(self):
        return self

    def _add_table(self, content, suffix):
        table_line = content.pop(0)
        if suffix == '.ext':
            table = ExtTable(''.join(content))
        elif suffix == '.phi':
            table = PhiTable(''.join(content))
        elif suffix == '.cov' or suffix == '.cor' or suffix == '.coi':
            table = CovTable(''.join(content))
        else:
            table = NONMEMTable(''.join(content))       # Fallback to non-specific table type
        m = re.match(r'TABLE NO.\s+(\d+)', table_line)   # This is guaranteed to match
        table.number = m.group(1)
        m = re.match(r'TABLE NO.\s+\d+: (.*?): (?:Goal Function=(.*): )?Problem=(\d+) '
                     r'Subproblem=(\d+) Superproblem1=(\d+) Iteration1=(\d+) Superproblem2=(\d+) '
                     r'Iteration2=(\d+)', table_line)
        if m:
            table.method = m.group(1)
            table.goal_function = m.group(2)
            table.problem = int(m.group(3))
            table.subproblem = int(m.group(4))
            table.superproblem1 = int(m.group(5))
            table.iteration1 = int(m.group(6))
            table.superproblem2 = int(m.group(7))
            table.iteration2 = int(m.group(8))
        self.tables.append(table)

    def __len__(self):
        return len(self.tables)

    @property
    def table(self, problem=1, subproblem=0, superproblem1=0, iteration1=0, superproblem2=0,
              iteration2=0):
        for t in self.tables:
            if t.problem == problem and t.subproblem == subproblem and \
                    t.superproblem1 == superproblem1 and t.iteration1 == iteration1 and \
                    t.superproblem2 == superproblem2 and t.iteration2 == iteration2:
                return t

    def table_no(self, table_number=1):
        for table in self.tables:
            if table.number == table_number:
                return table

    def __next__(self):
        if self._count >= len(self):
            raise StopIteration
        else:
            self._count += 1
            return self.tables[self._count - 1]


class NONMEMTable:
    '''A NONMEM output table.
    '''
    def __init__(self, content):
        self._df = pd.read_table(StringIO(content), sep=r'\s+', engine='python')

    @property
    def data_frame(self):
        return self._df

    @staticmethod
    def rename_index(df, ext=True):
        """ If columns rename also the row index
        """
        theta_labels = [x for x in df.columns if x.startswith('THETA')]
        omega_labels = [x for x in df.columns if x.startswith('OMEGA')]
        sigma_labels = [x for x in df.columns if x.startswith('SIGMA')]
        labels = theta_labels + omega_labels + sigma_labels
        if ext:
            labels = ['ITERATION'] + labels + ['OBJ']
        else:
            df = df.reindex(labels)
        df = df.reindex(labels, axis=1)
        df.columns = df.columns.str.replace(r'THETA(\d+)', r'THETA(\1)')
        if not ext:
            df.index = df.index.str.replace(r'THETA(\d+)', r'THETA(\1)')
        return df


class CovTable(NONMEMTable):
    @property
    def data_frame(self):
        df = self._df.copy(deep=True)
        df.set_index('NAME', inplace=True)
        df.index.name = None
        df = NONMEMTable.rename_index(df, ext=False)
        df = df.loc[(df != 0).any(axis=1), (df != 0).any(axis=0)]     # Remove FIX
        return df


class PhiTable(NONMEMTable):
    @property
    def data_frame(self):
        df = self._df.copy(deep=True)
        df.drop(columns=['SUBJECT_NO'], inplace=True)
        eta_col_names = [col for col in df if col.startswith('ETA')]
        # ETA column to be list of eta values for each ID
        df['ETA'] = df[eta_col_names].values.tolist()
        df.drop(columns=eta_col_names, inplace=True)
        etc_col_names = [col for col in df if col.startswith('ETC')]
        vals = df[etc_col_names].values
        matrix_array = [pharmpy.math.flattened_to_symmetric(x) for x in vals]
        # ETC column to be symmetric matrices for each ID
        df['ETC'] = matrix_array
        df.drop(columns=etc_col_names, inplace=True)
        return df


class ExtTable(NONMEMTable):
    @property
    def data_frame(self):
        df = self._df.copy(deep=True)
        df = NONMEMTable.rename_index(df)
        return df

    def _get_parameters(self, iteration, include_thetas=True):
        df = self.data_frame
        row = df.loc[df['ITERATION'] == iteration]
        del row['ITERATION']
        del row['OBJ']
        if not include_thetas:
            row = row[row.columns.drop(list(row.filter(regex='THETA')))]
        return row.squeeze()

    def _get_ofv(self, iteration):
        df = self.data_frame
        row = df.loc[df['ITERATION'] == iteration]
        row.squeeze()
        return row['OBJ'].squeeze()

    @property
    def final_parameter_estimates(self):
        '''Get the final parameter estimates as a Series
           A specific parameter estimate can be retreived as
           final_parameter_estimates['THETA1']
        '''
        return self._get_parameters(-1000000000)

    @property
    def standard_errors(self):
        '''Get the standard errors of the parameter estimates as a Series
        '''
        return self._get_parameters(-1000000001)

    @property
    def omega_sigma_stdcorr(self):
        '''Get the omegas and sigmas in standard deviation/correlation form
        '''
        return self._get_parameters(-1000000004, include_thetas=False)

    @property
    def omega_sigma_se_stdcorr(self):
        '''The standard error of the omegas and sigmas in stdcorr form
        '''
        return self._get_parameters(-1000000005, include_thetas=False)

    @property
    def fixed(self):
        '''What parameters were fixed?
        '''
        fix = self._get_parameters(-1000000006)
        return fix.apply(bool)

    @property
    def final_ofv(self):
        return self._get_ofv(-1000000000)

    @property
    def initial_ofv(self):
        return self._get_ofv(0)
