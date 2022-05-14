"""

Pattern for the intensity
        ;           ; Harm ;    Harm   ;    Harm    ; ;  Anh. ;            Anh.           ;    Anh.   ;     Anh.
 Region ; Exp Range ; Mode ;   Energy  ;  Activity  ; ; State ;           Assign          ;   Energy  ;   Activity
 III-12 ; 1234-1234 ;  123 ; 1234.6789 ;  1.345e-01 ; ;  1234 ; +1.34*|12(1),12(1),12(1)> ; 1234.6789 ;  1.345e-01
"""  # noqa

from cmath import exp
from curses import KEY_SPREVIOUS
from inspect import isframe
import os
import sys
from math import log, log10, pi, sqrt
import typing as tp

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

from estampes.data.physics import PHYSFACT, phys_fact
from estampes.parser import build_qlabel, DataFile
from estampes.base.spectrum import Spectrum
from estampes.tools.spec import convert_y

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('mathtext', fontset='stixsans')
mpl_loaded = True
# mpl.rcParams['backend'] = 'SVG'
# mpl.rcParams['backend.qt4'] = 'PyQt4'
plt.rcParams['font.family'] = 'sans-serif'


def build_mae_max(new_data: tp.Dict[str, tp.Any],
                  ref_data: tp.Dict[int, tp.Union[float, int]],
                  ax_labels: tp.Optional[tp.Dict[str, str]] = None,
                  canvas: tp.Optional[mpl.axes.Axes] = None,
                  csvfile: tp.Optional[str] = None,
                  quantity: str = 'freq'
                  ) -> tp.Optional[tp.Dict[str, tp.Any]]:
    """Builds and plots MAE and |MAX| errors.

    Computes mean absolute error and maximum unsigned error between
        new and reference data sets and plot them.
    The function can also return the detailed analysis.
    The structure of `new_data` depends on the type of quantity.

    Parameters
    ----------
    new_data
        New data.
    ref_data
        Reference data.
    ax_labels
        Labels to use in plots.
    canvas
        Matplotlib axis, where data will be displayed.
    csvfile
        Filename where error data will be saved.
    quantity
        Quantity of interest, used for the display and to identify the
        structure of `dnew` and `dref`.

    Returns
    -------
    dict, optional
        Errors, sorted by keys.

    Raises
    ------

    """
    def calc_diff(dnew: tp.Dict[int, tp.Union[float, int]],
                  dref: tp.Dict[int, tp.Union[float, int]]):
        diff = []
        for i in dnew:
            if isinstance(i, int) and i in dref:
                diff.append(dnew[i] - dref[i])
        return diff

    qty = quantity.lower().strip()
    dmax = {}
    dmae = {}
    if qty.startswith('freq'):
        for key in new_data:
            max_H = None
            mae_H = None
            max_A = None
            mae_A = None
            max_0 = None
            mae_0 = None
            if 'H' in new_data[key]:
                res = np.array(calc_diff(new_data[key]['H'], ref_data))
                num = len(res)
                max_H = np.max(np.abs(res))
                mae_H = np.sum(np.abs(res))/num
            if 'A' in new_data[key]:
                res = np.array(calc_diff(new_data[key]['A'], ref_data))
                num = len(res)
                max_A = np.max(np.abs(res))
                mae_A = np.sum(np.abs(res))/num
            if max_H is None and max_A is None:
                res = np.array(calc_diff(new_data[key], ref_data))
                num = len(res)
                max_0 = np.max(np.abs(res))
                mae_0 = np.sum(np.abs(res))/num
            if ax_labels is not None:
                if key in ax_labels:
                    dmax[key] = {'label': ax_labels[key]}
                    dmae[key] = {'label': ax_labels[key]}
            if key in dmax:
                if max_0 is None:
                    if max_H is not None:
                        dmax[key]['H'] = max_H
                        dmae[key]['H'] = mae_H
                    if max_A is not None:
                        dmax[key]['A'] = max_A
                        dmae[key]['A'] = mae_A
                else:
                    dmax[key]['value'] = max_0
                    dmae[key]['value'] = mae_0
            else:
                if max_0 is None:
                    dmax[key] = {}
                    dmae[key] = {}
                    if max_H is not None:
                        dmax[key]['H'] = max_H
                        dmae[key]['H'] = mae_H
                    if max_A is not None:
                        dmax[key]['A'] = max_A
                        dmae[key]['A'] = mae_A
                else:
                    dmax[key] = max_0
                    dmae[key] = mae_0
    else:
        raise KeyError('Unsupported quantity')

    if canvas is not None:
        plot_mae_max(dmax, dmae, canvas, quantity)

    return dmax, dmae


def plot_mae_max(dmax: tp.Dict[str, tp.Any],
                 dmae: tp.Dict[int, float],
                 canvas: tp.Optional[mpl.axes.Axes] = None,
                 quantity: str = 'freq'):
    """plots MAE and |MAX| errors.

    Plot the mean absolute error and maximum unsigned errors.
    The plot is chosen based on the quantity of interest.
    The structure of `dmax` and `dmae` depends on the type of quantity.

    Quantity == freq/freqH/freqA:
        Accepted structures:
        * dmax[label][level] = value
        * dmax[label] = value
        -> `level` can be 'H', 'A'

    The function support for all quantities a keyword, 'label', which
        overwrites the label to be used on display.  In this case, a
        single value (no sub-group) should be associated to the key
        'value'.

    Parameters
    ----------
    dmax
        Maximum unsigned error.
    dmae
        Mean unsigned error.
    canvas
        Matplotlib axis, where data will be displayed.
    quantity
        Quantity of interest, used for the display and to identify the
        structure of `dmax` and `dmae`.

    Raises
    ------

    """
    def autolabel(ax, rects, color=None):
        """Attach a text label above each bar, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.0f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", fontsize=12,
                        color=color,
                        ha='center', va='bottom', rotation=90)

    if dmax.keys() != dmae.keys():
        raise KeyError('Label inconsistency between |MAX| and MAE')

    qty = quantity.lower().strip()
    plot_data = {}
    datasets = []
    if qty.startswith('freq'):
        labels = []
        for key in dmae:
            if not isinstance(dmax[key], type(dmae[key])):
                msg = 'Inconsistenty between the data structures of |MAX| ' \
                    + 'and MAE'
                raise TypeError(msg)
            if isinstance(dmae[key], (float, int)):
                labels.append(key)
                if 'H' in plot_data or 'A' in plot_data:
                    msg = '''\
Data inconsistency in the |MAX|/MAE between methods for the frequencies.
Some have 'H'/'A' keys, others do not have this information.'''
                    raise KeyError(msg)
                if 0 not in plot_data:
                    use_H = qty != 'freqa'
                    plot_data[0] = {
                        'color': 'b' if use_H else 'r',
                        'label': r'$\omega$' if use_H else r'$\nu$',
                        'dmae': [],
                        'dmax': []
                    }
                try:
                    plot_data[0]['dmax'].append(float(dmax[key]))
                    plot_data[0]['dmae'].append(float(dmae[key]))
                except (ValueError, TypeError):
                    raise TypeError('Incorrect format for the error value.')
            elif isinstance(dmae[key], dict):
                if dmae[key].keys() != dmax[key].keys():
                    msg = 'Inconsistenty between the data structures of ' \
                        + '|MAX| and MAE fr label {}'.format(key)
                    raise TypeError(msg)
                data = dmae[key]
                if 'label' in data:
                    labels.append(data['label'])
                else:
                    labels.append(key)
                has_H = 'H' in data
                has_A = 'A' in data
                if has_H and qty in ('freq', 'freqh'):
                    if 0 in plot_data:
                        msg = '''\
Data inconsistency in the |MAX|/MAE between methods for the frequencies.
Some have 'H'/'A' keys, others do not have this information.'''
                        raise KeyError(msg)
                    if 'H' not in plot_data:
                        plot_data['H'] = {
                            'color': 'b',
                            'label': r'$\omega$',
                            'dmax': [],
                            'dmae': [],
                        }
                    try:
                        plot_data['H']['dmax'].append(float(dmax[key]['H']))
                        plot_data['H']['dmae'].append(float(dmae[key]['H']))
                    except (ValueError, TypeError):
                        msg = 'Incorrect format for the error value.'
                        raise TypeError(msg)
                if has_A and qty in ('freq', 'freqa'):
                    if 0 in plot_data:
                        msg = '''\
Data inconsistency in the |MAX|/MAE between methods for the frequencies.
Some have 'H'/'A' keys, others do not have this information.'''
                        raise KeyError(msg)
                    if 'A' not in plot_data:
                        plot_data['A'] = {
                            'color': 'r',
                            'label': r'$\nu$',
                            'dmax': [],
                            'dmae': [],
                        }
                    try:
                        plot_data['A']['dmax'].append(float(dmax[key]['A']))
                        plot_data['A']['dmae'].append(float(dmae[key]['A']))
                    except (ValueError, TypeError):
                        msg = 'Incorrect format for the error value.'
                        raise TypeError(msg)
                if 'value' in data:
                    if has_H or has_A:
                        msg = 'Data overlap: Both H/A and values defined ' \
                            + 'for {}'.format(key)
                        raise KeyError(msg)
                    if 'H' in plot_data or 'A' in plot_data:
                        msg = '''\
Data inconsistency in the |MAX|/MAE between methods for the frequencies.
Some have 'H'/'A' keys, others do not have this information.'''
                        raise KeyError(msg)
                    if 0 not in plot_data:
                        use_H = qty != 'freqa'
                        plot_data[0] = {
                            'color': 'b' if use_H else 'r',
                            'label': r'$\omega$' if use_H else r'$\nu$',
                            'dmae': [],
                            'dmax': []
                        }
                    try:
                        plot_data[0]['dmax'].append(float(dmax[key]))
                        plot_data[0]['dmae'].append(float(dmae[key]))
                    except (ValueError, TypeError):
                        msg = 'Incorrect format for the error value.'
                        raise TypeError(msg)
        if 0 in plot_data:
            datasets.append(plot_data[0])
        else:
            if 'H' in plot_data:
                datasets.append(plot_data['H'])
            if 'A' in plot_data:
                datasets.append(plot_data['A'])
        ylabel = r'Error wrt Experiment (cm$^{-1}$)'
    else:
        raise KeyError('Unsupported quantity')

    # Color codes:
    # b: blue (light/dark)
    # r: red (pink/red)
    c_MAE = {'b': '#1317ff', 'r': '#bd2626'}
    c_MAX = {'b': '#6baff0', 'r': '#ff86db'}

    ind = np.arange(len(labels))
    if len(datasets) == 2:
        width = 0.2

        for i in range(len(datasets)):
            bar_data = []
            bar_data.append(
                canvas.bar(ind+(4*i-1)*width/2, datasets[i]['dmax'], width,
                           color=c_MAX[datasets[i]['color']]))
            bar_data.append(
                canvas.bar(ind+(4*i+1)*width/2, datasets[i]['dmae'], width,
                           color=c_MAE[datasets[i]['color']]))
            autolabel(canvas, bar_data[0], c_MAX[datasets[i]['color']])
            autolabel(canvas, bar_data[1], c_MAE[datasets[i]['color']])
        # ymax = max(*dmue_h, *amax_h, *amax_a, *dmue_a)
        # ax1.set_ylim(top=ymax*1.1)
        # ymax = ax1.get_ylim()[1]
        # ax1.set_ylim(top=ymax+10)
        canvas.set_ylabel(ylabel,fontsize=12)
        canvas.grid(axis='x', lw=.2, ls='--', c='gray')
        canvas.set_xticks(ind+width)
        lab_meth = [item.replace(r'_', r'\_') for item in labels]
        canvas.set_xticklabels(tuple(lab_meth), fontsize=12, rotation=20,
                               ha="right")
        canvas.legend((r'$|$MAX$|$ ({})'.format(datasets[0]['label']),
                       r'MAE ({})'.format(datasets[0]['label']),
                       r'$|$MAX$|$ ({})'.format(datasets[1]['label']),
                       r'MAE ({})'.format(datasets[1]['label'])),
                      loc='best',ncol=2)
    else:
        width = 0.4

        bar_data = []
        bar_data.append(
            canvas.bar(ind-width, datasets[0]['dmax'], width,
                       color=c_MAX[datasets[0]['color']]))
        bar_data.append(
            canvas.bar(ind+width, datasets[0]['dmae'], width,
                       color=c_MAE[datasets[0]['color']]))
        autolabel(canvas, bar_data[0], c_MAX[datasets[0]['color']])
        autolabel(canvas, bar_data[1], c_MAE[datasets[0]['color']])

        canvas.legend((r'MAE ({})'.format(datasets[0]['label']),
                       r'$|$MAX$|$ ({})'.format(datasets[0]['label'])),ncols=2)
        canvas.set_ylabel(ylabel)
        canvas.grid(axis='x', lw=.2, ls='--', c='gray')
        canvas.set_xticks(ind)

        lab_meth = [item.replace(r'_', r'\_') for item in labels]
        canvas.set_xticklabels(tuple(lab_meth), fontsize=12, rotation=20,
                               ha="right")


if __name__ == '__main__':

    all_files = {
        'freq_exp': 'li_energy.txt',
        'B2PD/ATZ': 'li_MO_b01D_GV.log',
        'B2PD/JNTZ': 'li_MO_b02D_GV.log',
        'RDSD/JNTZ': 'li_MO_b18D_GV.log',
        'B3LD/JNTZ': 'li_MO_b17D_GV.log',
        'PW6D/JNTZ': 'li_MO_b23D_GV.log',
        'B3PD/JNTZ': 'li_MO_b24D_GV.log',
        'B3PD/JLTZ': 'li_MO_b31D_GV.log',
        'B2PD/JNTZ//B3PD/JNTZ': 'd_MO_b02Db24D.log',
        'RDSD/JNTZ//B3PD/JNTZ': 'e_MO_b18Db24D.log',
        'CCSD/ATZ//B3PD/JNTZ': 'f_MO_b24D_CCSD.log',
        '3nq': 'b1_MO_b18b24DR1_f3_3q_j19N.log',
        'peak_zones': 'peaks_1.txt',
        'exp_Raman': 'MO_IRaman.txt',
        'exp_ROA': 'MO_IROA.txt',
    }
    nvib = 24
    Rincfrq = {
        'H': '18796.99',
        'A': '18796.9925'
    }
    # Setups refer to the Raman setups.  We need this flag because the
    #    label is slightly different between the two.
    setups = {
        'H': 'SCP(180)u',
        'A': 'SCP(180)',
    }

    build_fig_MAE = True
    build_tab_assign = False
    build_main_spec = False
    build_zone_spec = False 
    build_zone_assign = False
    build_spec = build_main_spec or build_zone_spec or build_zone_assign
    do_spec = build_tab_assign or build_spec
    img_ext = '.pdf'

    assign_data = {}

    if do_spec:
        peaks = {}
        with open(all_files['peak_zones']) as fobj:
            for line in fobj:
                if line.strip() and not line.startswith('Peaks'):
                    key, rng = line.split(':')
                    peaks[key] = tuple([int(i) for i in rng.split('-')])

        dkeys_spc = {
            'assign': build_qlabel('vtrans', level='A'),
            'freqA': build_qlabel('vlevel', level='A'),
            'freqH': build_qlabel('vlevel', level='H'),
            'transA': build_qlabel('vtrans', level='A'),
            'transH': build_qlabel('vtrans', level='H'),
            'RSA': build_qlabel('RamAct', 'dynamic', level='A'),
            'RSH': build_qlabel('RamAct', 'dynamic', level='H'),
            'ROAA': build_qlabel('ROAAct', 'dynamic', level='A'),
            'ROAH': build_qlabel('ROAAct', 'dynamic', level='H'),
            'coef': build_qlabel('vptdat', 'CICoef'),
        }
        fname = all_files['3nq']
        print('Parsing file: {}'.format(fname))
        dfile = DataFile(fname, filetype='glog')
        data_spc = dfile.get_data(*dkeys_spc.values(), error_noqty=False)

    if build_fig_MAE:
        to_include = (
            'B2PD/JNTZ',
            'RDSD/JNTZ',
            'B3PD/JNTZ',
            'B3LD/JNTZ',
            'PW6D/JNTZ',
            'B2PD/JNTZ//B3PD/JNTZ',
            'RDSD/JNTZ//B3PD/JNTZ',
            'CCSD/ATZ//B3PD/JNTZ'
        )
        modes_to_exclude = (20,)
        imgfile = 'fig_HA' + img_ext
        # For debugging purposes
        dump_file = True  # print table all the frequencies
        # Fix order of anharmonic CH stretching region, since higher mixing
        fix_CH_order = True  # reorganize with increasing order
        # Only consider below 2800
        below_2800 = True
        if below_2800:
            imgfile = '_below_2800'.join(os.path.splitext(imgfile))
        # Get experimental data
        ref_data = {}
        with open(all_files['freq_exp']) as fobj:
            for line in fobj:
                cols = line.split()
                if len(cols) == 2:
                    if int(cols[0]) not in modes_to_exclude:
                        ref_data[int(cols[0])] = float(cols[1])

        calc_data = {}
        dkeys = {
            # 'assign': build_qlabel('vtrans', level='A'),
            'freqA': build_qlabel('vlevel', level='A'),
            'freqH': build_qlabel('vlevel', level='H'),
            'transA': build_qlabel('vtrans', level='A'),
            'transH': build_qlabel('vtrans', level='H'),
            'coef': build_qlabel('vptdat', 'CICoef'),
        }
        dkeys_H = {
            'freqH': build_qlabel('vlevel', level='H'),
        }
        for key in to_include:
            fname = all_files[key]
            print('Parsing file: {}'.format(fname))
            if dump_file:
                fblock = os.path.splitext(fname)[0]
            dfile = DataFile(fname, filetype='glog')
            data = dfile.get_data(*dkeys.values(), error_noqty=False)
            # 'CC' ignored since no log file available
            hybrid = '//' in key and not key.startswith('CC')
            if hybrid:
                dfile_H = DataFile(all_files[key.split('//')[0]],
                                   filetype='glog')
                data_H = dfile_H.get_data(*dkeys_H.values(),
                                          error_noqty=False)
            if dump_file:
                fmt_A = ' {:3d} ; {:5.3f} ; {:9.4f} ; {:9.4f}'
                fmt_NA = ' {:3d} ; {:5.3f} ; {:9.4f};    ---'
                dump_data = {}
            calc_data[key] = {'H': {}, 'A': {}}
            if data[dkeys['coef']] is not None:
                states = data[dkeys['coef']]
                for i in states:
                    if isinstance(i, int):
                        coefs = sorted(states[i], key=lambda x: x[0]**2,
                                       reverse=True)
                        if len(coefs[0][1]) == 1 and coefs[0][1][0][1] == 1:
                            mode = coefs[0][1][0][0]
                            if mode not in modes_to_exclude:
                                coef = coefs[0][0]**2
                                enA = data[dkeys['freqA']][i]
                                if hybrid:
                                    enH = data_H[dkeys_H['freqH']][mode]
                                else:
                                    enH = data[dkeys['freqH']][mode]
                                if not below_2800 or enA <= 2800:
                                    if dump_file:
                                        dump_data[mode] = fmt_A.format(
                                            mode, coef, enH, enA)
                                    calc_data[key]['H'][mode] = enH
                                    calc_data[key]['A'][mode] = enA
            else:
                for i in data[dkeys['transA']]:
                    fsta = data[dkeys['transA']][i][1]
                    if len(fsta) == 1 and fsta[0][1] == 1:
                        mode = fsta[0][0]
                        if mode not in modes_to_exclude:
                            enA = data[dkeys['freqA']][i]
                            if hybrid:
                                enH = data_H[dkeys_H['freqH']][mode]
                            else:
                                enH = data[dkeys['freqH']][mode]
                            if not below_2800 or enA <= 2800:
                                calc_data[key]['H'][mode] = enH
                                calc_data[key]['A'][mode] = enA
                                if dump_file:
                                    dump_data[mode] = fmt_A.format(
                                        mode, 1.0, enH, enA)
            n = len(calc_data[key]['A'])
            if n < nvib:
                fmt = 'Warning: Missing modes. Only {} assigned vs {}'
                print(fmt.format(n, nvib))
            # We assume that the last mode is not passive/inactive.
            if dump_file:
                nmodes = max(dump_data.keys())
                for i in range(1, nmodes+1):
                    if i not in dump_data:
                        dump_data[i] = fmt_NA.format(
                            i, 1.0, data[dkeys['freqH']][i])
                with open('freq_{}_anh.csv'.format(fblock), 'w') as fobj:
                    # fobj = sys.stdout
                    print('Mode ; Coeff ;    E(H)   ;    E(A)', file=fobj)
                    for i in range(1, nmodes+1):
                        print(dump_data[i], file=fobj)

        if fix_CH_order:
            for key in calc_data:
                indxs = []
                freqs = []
                for i, value in calc_data[key]['A'].items():
                    if value > 2800:
                        indxs.append(i)
                        freqs.append(value)
                indxs.sort()
                freqs.sort()
                for i in range(len(indxs)):
                    calc_data[key]['A'][indxs[i]] = freqs[i]
            imgfile = '_CH_reord'.join(os.path.splitext(imgfile))
        figsize = (7, 6)
        fig = plt.figure(1, figsize=figsize, dpi=100)
        ax1 = fig.add_subplot(111)
        dmax, dmae = build_mae_max(calc_data, ref_data, canvas=ax1)
        if below_2800:
            plt.ylim((0, 70))
        else:
            plt.ylim((0, 200))
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.001)
        plt.savefig(imgfile, bbox_inches='tight')
        plt.show()
        plt.close(1)

    if build_tab_assign or build_zone_assign:
        def Econv(val: float) -> float:
            return val*scale_freq

        def Iconv(val: float, omega: float) -> float:
            return val*phys_fact('mwq2q')**2*PHYSFACT.bohr2ang**2/(2.*omega)

        params = {
            # 'RS': {
            #     'min': {
            #         'I': 1e-1,
            #         'II': 2e-3,
            #         'III': 1e-1,
            #         'IV': 2e-3,
            #     },
            #     'out': 'Raman',
            # },
            'ROA': {
                'min': {
                    'I': 2e-5,
                    'II': 7e-7,
                    'III': 2e-5,
                    'IV': 7e-7,
                },
                'out': 'ROA',
            },
        }
        fmt_zone = '{:>6s} ; {:4d}-{:4d}'
        nul_zone = '       ;          '
        fmt_harm = '{:4d} ; {:9.4f} ; {:10.3e} ; {:10.3e} '
        nul_harm = '     ;           ;            ;            '
        fmt_anh0 = '{:5d} ; {:+5.2f}*{:<19s} ; {:9.4f} ; {:10.3e} ; {:10.3e} ' 
        fmt_anh1 = '      ; {:+5.2f}*{:<19s} ;         ;            ; '
        nul_anh = '      ;                           ;           ;'
        fmt_full = ' {} ; {} ; {}\n'
        ref_incfrq = int(float(Rincfrq['H']))
        scale_freq = 0.967
        # Include all harmonic modes, regardless of their intensity
        include_all_fund = True

        states = data_spc[dkeys_spc['coef']]
        # data_spc is a data structure,
        # dkeys_spc is a dictionary for the structure
        # dkeys_coef contains the VPT2 calculations
        # Eh is the energy of harmonic
        # Eh is a dictionary, because of Eh has 25 components, so 1 needs to be
        #    subtracted
        Eh = data_spc[dkeys_spc['freqH']]
        maxstatesH = len(Eh) - 1
        # Ea is the energy of anharmonic
        Ea = data_spc[dkeys_spc['freqA']]

        maxstatesA = len(Ea) - 1
        for spec in params:
            RS_lm={
                    'I': 1e-1,
                    'II': 3e-3,
                    'III': 1e-1,
                    'IV': 3e-3,
                }
            assign_data[spec] = {}
            with (open('data_{}_assign.csv'.format(params[spec]['out']), 'w')
                  as fobj):
                fobj.write('''\
        ;           ; Harm  ;    Harm   ;   Harm ROA  ;   Harm Ram ;\
  Anh. ;            Anh.           ;  Anh. ROA ;     Anh. Ram
 Region ; Exp Range ; Mode  ;   Energy  ;  Harm ROA  ;   HRaman   ;\
 State ;           Assign          ;   Energy  ;   IROA      ;  IRaman
''')
                key_spc = dkeys_spc['{}H'.format(spec)]
                key_spc_HRM=dkeys_spc['{}H'.format('RS')]
                key_frq = None
                # Find the correct keyword for the incident light of interest
                # The idea is that there will be no 2 laser beams within 1 cm-1
                # So se can simply look at the integer part.
                for key in data_spc[key_spc].keys():
                    if key.split('.')[0] == str(ref_incfrq):
                        key_frq = key
                        break
                if key_frq is None:
                    fmt = 'ERROR: Did not find reference incident frequency ' \
                        + '{} for harmonic.'
                    print(fmt.format(ref_incfrq))
                    sys.exit()
                Ih = data_spc[key_spc][key_frq][setups['H']]
                Ih_HRM = data_spc[key_spc_HRM][key_frq][setups['H']]
                
                key_spc = dkeys_spc['{}A'.format(spec)]
                key_spc_RM = dkeys_spc['{}A'.format('RS')]
                key_frq = None

                for key in data_spc[key_spc].keys():
                    if key.split('.')[0] == str(ref_incfrq):
                        key_frq = key
                        break
                if key_frq is None:
                    fmt = 'ERROR: Did not find reference incident frequency ' \
                        + '{} for VPT2.'
                    print(fmt.format(ref_incfrq))
                    sys.exit()
                Ia = data_spc[key_spc][key_frq][setups['A']]
                Ia_RM=data_spc[key_spc_RM][key_frq][setups['A']]

                #Iaram =   
                ind_H = 1
                ind_A = 1
                # We may wish to include all fundamentals
                # This is done by setting Imin to 0 for the harmonic part.
                if include_all_fund:
                    IminH = 0.0
                else:
                    IminH = None
                for zone in peaks:
                    statesH = []
                    statesA = []
                    Emin, Emax = peaks[zone]
                    reg = zone.split('-')[0]
                    if reg not in assign_data[spec]:
                        assign_data[spec][reg] = {'H': {}, 'A': {}}
                    Imin = params[spec]['min'][reg]
                    Imin_RM=RS_lm[reg]

                    if IminH is None:
                        IminH = Imin
                    # We take advantage of the fact that the states are ordered
                    #   by increasing order.  Otherwise, need to make a more
                    #   generic loop.
                    if ind_H <= maxstatesH:
                        while (E := Econv(Eh[ind_H])) <= Emax:
                            if E >= Emin:
                                if abs(Iconv(Ih[ind_H], Eh[ind_H])) > IminH:
                                    statesH.append(ind_H)
                                    assign_data[spec][reg]['H'][ind_H] = \
                                        (Eh[ind_H],
                                         Iconv(Ih[ind_H], Eh[ind_H]))
                            ind_H += 1
                            if ind_H > maxstatesH:
                                break
                    if ind_A <= maxstatesA:
                        while Ea[ind_A] <= Emax:
                            if Ea[ind_A] >= Emin:
                                print (states[ind_A])
                                print (states[ind_A][0][1][0][1])
                                if ((
                                     #len(states[ind_A]) == 1 and
                                     len(states[ind_A][0][1]) == 1 and
                                     states[ind_A][0][0] >= 0.5 and
                                     states[ind_A][0][1][0][1] == 1)
                                        or abs(Ia[ind_A]) > Imin 
                                        or abs(Ia_RM[ind_A])>Imin_RM):
                                    
                                    statesA.append(ind_A)
                                    assign_data[spec][reg]['A'][ind_A] = \
                                        (Ea[ind_A], Ia[ind_A])

                            
                            ind_A += 1
                            if ind_A > maxstatesA:
                                break
                    nstatesH = len(statesH)
                    nstatesA = len(statesA)
                    for i in range(max(nstatesH, nstatesA)):
                        if i == 0:
                            txt_zone = fmt_zone.format(zone, Emin, Emax)
                        else:
                            txt_zone = nul_zone
                        if i < nstatesH:
                            ii = statesH[i]

                            txt_harm = fmt_harm.format(
                                ii, Econv(Eh[ii])/scale_freq, Iconv(Ih[ii], Eh[ii]/scale_freq),Iconv(Ih_HRM[ii],Eh[ii]/scale_freq))
                        else:
                            txt_harm = nul_harm
                        if i < nstatesA:
                            ii = statesA[i]
                            coefs = [item
                                     for item in sorted(states[ii],
                                                        key=lambda x: x[0]**2,
                                                        reverse=True)
                                     if item[0]**2 > 0.1]
                            overlap = '|' \
                                + ','.join(['{}({})'.format(*item)
                                            for item in coefs[0][1]]) \
                                + '>'
                            txt_anh = fmt_anh0.format(ii, coefs[0][0], overlap,
                                                      Ea[ii], Ia[ii],Ia_RM[ii])
                            fobj.write(fmt_full.format(txt_zone, txt_harm,
                                                       txt_anh))
                            for coef in coefs[1:]:
                                overlap = '|' \
                                    + ','.join(['{}({})'.format(*item)
                                                for item in coef[1]]) \
                                    + '>'
                                txt_anh = fmt_anh1.format(coef[0], overlap)
                                fobj.write(fmt_full.format(nul_zone, nul_harm,
                                                           txt_anh))
                        else:
                            fobj.write(fmt_full.format(txt_zone, txt_harm,
                                                       nul_anh))

    if build_spec:
        c_A = '#ef2b2b'
        c_H = '#2b86ef'
        fname = all_files['3nq']
        # We list all HWHMs of interest
        hwhms = (5, 10)
        # Filenames
        # * for CSV file
        tmpl_fname = 'spec_{lvl}_{spc}_G{hw:02d}.csv'
        # * for figures
        imgfile_main = 'fig_spec_full' + img_ext
        imgfile_zones = 'fig_spec_zone_{}' + img_ext
        imgfile_assign = 'fig_spec_assign_{}' + img_ext

        # Legend for Y axis
        fmt_I_RS = '$I$ / 10$^{{{}}}$ cm$^3$ mol$^{{-1}}$ sr$^{{-1}}$'
        fmt_I_ROA = '$\\Delta I$ / 10$^{{{}}}$ cm$^3$ mol$^{{-1}}$ sr$^{{-1}}$'
        # Parameters for the spectral zones
        exp_rng = {
            'I': [None, None],
            'II': [None, None],
            'III': [None, None],
            'IV': [None, None],
        }
        for key in peaks:
            rng = key.split('-')[0]
            emin, emax = peaks[key]
            if exp_rng[rng][0] is None:
                exp_rng[rng][0] = emin
            else:
                if exp_rng[rng][0] >= emin:
                    exp_rng[rng][0] = emin
            if exp_rng[rng][1] is None:
                exp_rng[rng][1] = emax
            else:
                if exp_rng[rng][1] <= emax:
                    exp_rng[rng][1] = emax

        Yrang = {
            'I_ram': [0.0, 2.8, 1e-7, [0, 1, 2]],
            'I_roa': [-3, 3, 1e-9, [-2, 0, 2]],
            'II_ram': [0, 3, 1e-9, [0, 1, 2]],
            'II_roa': [-5, 5, 1e-11, [-4, 0, 4]],
            'III_roa': [-1.9, 1.9, 1e-9, [-1, 0, 1]],
            'III_ram': [0, 7, 1e-7, [0, 3, 6]],
            'IV_ram': [0, 1.5, 1e-9, [0, 0.5, 1]],
            'IV_roa': [-1.7, 1.7, 1e-11, [-1, 0, 1]],
            'main_ram': [0, 8, 1e-7, [0, 3, 6]],
            'main_roa': [-3.4, 4, 1e-9, [-2, 0, 2]]
        }

        specs = {
            'RS': {'H': {}, 'A': {}},
            'ROA': {'H': {}, 'A': {}},
        }
        reffiles_ok = [os.path.exists(tmpl_fname.format(lvl='H', spc='RS',
                                                        hw=hw))
                       for hw in hwhms]
        if not all(reffiles_ok):
            for lvl in ('H', 'A'):
                for spc in specs.keys():
                    spec = Spectrum(
                        all_files['3nq'], spc, lvl, ftype='glog',
                        incfrq=Rincfrq[lvl], setup=setups[lvl])
                    for hwhm in hwhms:
                        spec.set_broadening(
                            hwhm, 'gaussian', 'I:cm3/mol/sr', 2, 10, 5000)
                        specs[spc][lvl][hwhm] = {
                            'x': np.copy(spec.xaxis),
                            'y': np.copy(spec.yaxis)
                        }
                        np.savetxt(
                            tmpl_fname.format(lvl=lvl, spc=spc, hw=hwhm),
                            np.transpose((spec.xaxis, spec.yaxis)),
                            fmt=('%10.4f', '%20.8e'))
        else:
            for lvl in ('H', 'A'):
                for spc in specs.keys():
                    for hwhm in hwhms:
                        fname = tmpl_fname.format(lvl=lvl, spc=spc, hw=hwhm)
                        print('Parsing file: {}'.format(fname))
                        specs[spc][lvl][hwhm] = np.genfromtxt(
                            fname, names=['x', 'y'])

        # Let us build now the full spectrum by combining the different hwhms
        # We use the where statement to choose the y, the x axis being always
        #    the same
        for lvl in ('H', 'A'):
            for spc in specs.keys():
                specs[spc][lvl]['x'] = np.where(
                    specs[spc][lvl][hwhms[0]]['x'] < exp_rng['III'][0],
                    specs[spc][lvl][hwhms[0]]['x'],
                    specs[spc][lvl][hwhms[0]]['x']-18)
                specs[spc][lvl]['y'] = np.where(
                    specs[spc][lvl][hwhms[0]]['x'] < exp_rng['III'][0],
                    specs[spc][lvl][hwhms[0]]['y'],
                    specs[spc][lvl][hwhms[1]]['y']
                )

        # Now extract and convert the experimental data
        spec_exp = {
            'RS': np.genfromtxt(all_files['exp_Raman'], names=['x', 'y']),
            'ROA': np.genfromtxt(all_files['exp_ROA'], names=['x', 'y'])
        }
        spec_exp['RS']['y'] /= 1.8e14
        spec_exp['ROA']['y'] /= -1.8e12

        if build_main_spec:
            fig = plt.figure(figsize=(6, 6), dpi=300)
            subp = fig.subplots(2, 1)

            xmin, xmax = exp_rng['I'][0], exp_rng['IV'][1]
            subp[0].plot(spec_exp['RS']['x'],
                         spec_exp['RS']['y']/Yrang['main_ram'][2],
                         c='black', label='Exp.')
            subp[0].plot(specs['RS']['H']['x'],
                         specs['RS']['H']['y']/Yrang['main_ram'][2]+0.4*Yrang['main_ram'][1],
                         c=c_H, label='Harm.')
            subp[0].plot(specs['RS']['A']['x'],
                         specs['RS']['A']['y']/Yrang['main_ram'][2]-0.4*Yrang['main_ram'][1],
                         c=c_A, label='Anh.')
            subp[1].plot(spec_exp['ROA']['x'],
                         spec_exp['ROA']['y']/Yrang['main_roa'][2],
                         c='black', label='Exp.')
            subp[1].plot(specs['ROA']['H']['x'],
                         specs['ROA']['H']['y']/Yrang['main_roa'][2]+0.8*Yrang['main_roa'][1],
                         c=c_H, label='Harm.')
            subp[1].plot(specs['ROA']['A']['x'],
                         specs['ROA']['A']['y']/Yrang['main_roa'][2]-0.8*Yrang['main_roa'][1],
                         c=c_A, label='Anh.')
            subp[0].set_xlim(xmin, xmax)
            subp[1].set_xlim(xmin, xmax)
            subp[0].set_ylim(bottom=0.0)
            subp[0].set_ylim(-0.5*Yrang['main_ram'][1],1.2*Yrang['main_ram'][1])
            subp[1].set_ylim(-1.6*Yrang['main_roa'][1],1.7*Yrang['main_roa'][1])
            plt.subplots_adjust(hspace=0)

            for key in exp_rng:
                if key != 'IV':
                    subp[0].axvline(exp_rng[key][1], ls='--', c='#a3a3a3',
                                    lw=.8)
                    subp[1].axvline(exp_rng[key][1], ls='--', c='#a3a3a3',
                                    lw=.8)
                xval = (exp_rng[key][0] + exp_rng[key][1])/2
                xrel = (xval - xmin)/(xmax - xmin)
                text = '({})'.format(key)
                # @QY: you use 12 as fontsize, but 'large' is more generic and
                #    will work on configurations with a different density.
                #    If you still prefer, use 12 below.
                # subp[0].text(xrel, 0.8, text, ha='center',
                #              fontsize='large', transform=subp[0].transAxes)
                subp[1].text(xrel, 0.87, text, ha='center',
                             fontsize='large', transform=subp[1].transAxes)
            subp[1].set_xlabel('Wavenumbers / cm$^{-1}$', fontsize=12)
            subp[0].set_ylabel(
                fmt_I_RS.format(int(log10(Yrang['main_ram'][2]))))
            subp[1].set_ylabel(
                fmt_I_ROA.format(int(log10(Yrang['main_roa'][2]))))
            # subp[0].legend()
            subp[0].legend(loc='upper left', ncol=3)
            # plt.tight_layout()
            plt.savefig(imgfile_main)
            plt.show()

        elif build_zone_spec:
            for rng in exp_rng:
                fig, subp = plt.subplots(2, 1)
                subp[0].set_xlim(*exp_rng[rng])
                subp[1].set_xlim(*exp_rng[rng])
                xmin, xmax = exp_rng[rng]

                yexp_off = -0.153e-8  # Y offset for some experimental data
                if rng in ('III', 'IV'):
                    # Note that hw may have to be updated if the structure of
                    #    hwhms change.
                    hw = hwhms[-1]
                    xcal_off = -18  # X offset for calculated data
                else:
                    hw = hwhms[0]
                    xcal_off = 0

                subp[0].plot(
                    spec_exp['RS']['x'],
                    (spec_exp['RS']['y']+yexp_off)/Yrang[rng+'_ram'][2],
                    c='black', label='Exp.')
                subp[0].plot(
                    specs['RS']['H'][hw]['x'] + xcal_off,
                    specs['RS']['H'][hw]['y']/Yrang[rng+'_ram'][2],
                    c=c_H, label='Harm.')
                subp[0].plot(
                    specs['RS']['A'][hw]['x'] + xcal_off,
                    specs['RS']['A'][hw]['y']/Yrang[rng+'_ram'][2],
                    c=c_A, label='Anh.')
                subp[1].plot(
                    spec_exp['ROA']['x'],
                    spec_exp['ROA']['y']/Yrang[rng+'_roa'][2],
                    c='black', label='Exp.')
                subp[1].plot(
                    specs['ROA']['H'][hw]['x'] + xcal_off,
                    specs['ROA']['H'][hw]['y']/Yrang[rng+'_roa'][2],
                    c=c_H, label='Harm.')
                subp[1].plot(
                    specs['ROA']['A'][hw]['x'] + xcal_off,
                    specs['ROA']['A'][hw]['y']/Yrang[rng+'_roa'][2],
                    c=c_A, label='Anh.')

                max_x = 0.0
                for key, val in peaks.items():
                    if key.startswith('{}-'.format(rng)):
                        text = '{}'.format(key.split('-')[1])
                        if int(text) % 2 == 0:
                            subp[0].axvspan(
                                val[0], val[1], facecolor='g', alpha=0.1)
                            subp[1].axvspan(
                                val[0], val[1], facecolor='g', alpha=0.1)

                        xval = (val[0] + val[1])/2
                        xrel = (xval - xmin)/(xmax - xmin)
                        # subp[0].text(xrel, 0.8, text, ha='center',
                        #              fontsize='large',
                        #              transform=subp[0].transAxes)
                        text = '({})'.format(key.split('-')[1])
                        subp[1].text(xrel, 0.9, text, ha='center',
                                     fontsize=12,
                                     transform=subp[1].transAxes)
                    max_x = val[1]

                plt.subplots_adjust(hspace=0)
                subp[0].set_ylim(Yrang[rng+'_ram'][0], Yrang[rng+'_ram'][1])
                subp[1].set_ylim(Yrang[rng+'_roa'][0], Yrang[rng+'_roa'][1])
                subp[1].set_xlabel('Wavenumbers / cm$^{-1}$', fontsize=12)
                # format the y number

                subp[0].yaxis.set_ticks(Yrang[rng+'_ram'][3])
                subp[1].yaxis.set_ticks(Yrang[rng+'_roa'][3])
                subp[0].legend(loc='upper left', ncol=3)
                # subp[0].xaxis.set_ticks([])
                subp[0].tick_params(axis='x', direction='in', which='both',
                                    top=True, labeltop=True)
                subp[0].grid(axis='x', linestyle='--', linewidth='0.2')
                subp[1].grid(axis='x', linestyle='--', linewidth='0.2')
                subp[0].set_ylabel(
                    fmt_I_RS.format(int(log10(Yrang[rng+'_ram'][2]))))
                subp[1].set_ylabel(
                    fmt_I_ROA.format(int(log10(Yrang[rng+'_roa'][2]))))

                plt.savefig(imgfile_zones.format(rng))
            plt.show()

        elif build_zone_assign:
            y_RS = convert_y('RS', 'I:cm3/mol/sr', 'RA:Ang6',
                             incfreq=float(Rincfrq['A']))
            y_ROA = convert_y('ROA', 'I:cm3/mol/sr', 'ROA:Ang6',
                              incfreq=float(Rincfrq['A']))

            for rng in exp_rng:
                fig, subp = plt.subplots(2, 1)
                subp[0].set_xlim(*exp_rng[rng])
                subp[1].set_xlim(*exp_rng[rng])
                xmin, xmax = exp_rng[rng]

                yexp_off = -0.153e-8  # Y offset for some experimental data
                if rng in ('III', 'IV'):
                    # Note that hw may have to be updated if the structure of
                    #    hwhms change.
                    hw = hwhms[-1]
                    xcal_off = -18  # X offset for calculated data
                else:
                    hw = hwhms[0]
                    xcal_off = 0

                subp[0].plot(
                    spec_exp['RS']['x'],
                    (spec_exp['RS']['y']+yexp_off)/Yrang[rng+'_ram'][2],
                    c='black', label='Exp.')
                subp[0].plot(
                    specs['RS']['H'][hw]['x'] + xcal_off,
                    specs['RS']['H'][hw]['y']/Yrang[rng+'_ram'][2],
                    c=c_H, label='Harm.')
                subp[0].plot(
                    specs['RS']['A'][hw]['x'] + xcal_off,
                    specs['RS']['A'][hw]['y']/Yrang[rng+'_ram'][2],
                    c=c_A, label='Anh.')
                subp[1].plot(
                    spec_exp['ROA']['x'],
                    spec_exp['ROA']['y']/Yrang[rng+'_roa'][2],
                    c='black', label='Exp.')
                subp[1].plot(
                    specs['ROA']['H'][hw]['x'] + xcal_off,
                    specs['ROA']['H'][hw]['y']/Yrang[rng+'_roa'][2],
                    c=c_H, label='Harm.')
                subp[1].plot(
                    specs['ROA']['A'][hw]['x'] + xcal_off,
                    specs['ROA']['A'][hw]['y']/Yrang[rng+'_roa'][2],
                    c=c_A, label='Anh.')

                # Add harmonic and anharmonic assignment labels
                # We compute an offset based on the overall height of the
                #   spectrum to have a visible shift
                # It may be useful to offset ymax to let the labels be visible
                height = Yrang[rng+'_ram'][1] - Yrang[rng+'_ram'][0]
                yoffset = height * .05
                yscale = y_RS[0]/(sqrt(hw**2/log(2)*pi)*Yrang[rng+'_ram'][2])
                xfunc = y_RS[1]
                for state, vals in assign_data['RS'][rng]['A'].items():
                    y = yscale*vals[1]*xfunc(vals[0]) + yoffset
                    subp[0].text(vals[0]+xcal_off, y, str(state), rotation=90,
                                 ha='center', va='bottom',
                                 fontsize='small', color=c_A)
                for state, vals in assign_data['RS'][rng]['H'].items():
                    y = yscale*vals[1]*xfunc(vals[0]) + yoffset
                    subp[0].text(vals[0]+xcal_off, y, str(state), rotation=90,
                                 ha='center', va='bottom',
                                 fontsize='small', color=c_H)

                height = Yrang[rng+'_roa'][1] - Yrang[rng+'_roa'][0]
                yoffset = height*.05
                yscale = y_ROA[0]/(sqrt(hw**2/log(2)*pi)*Yrang[rng+'_roa'][2])
                xfunc = y_ROA[1]
                for state, vals in assign_data['ROA'][rng]['A'].items():
                    y = yscale*vals[1]*xfunc(vals[0])
                    if y > 0.0:
                        y += yoffset
                        align = 'bottom'
                    else:
                        y -= yoffset
                        align = 'top'
                    subp[1].text(vals[0]+xcal_off, y, str(state), rotation=90,
                                 ha='center', va=align,
                                 fontsize='small', color=c_A)
                for state, vals in assign_data['ROA'][rng]['H'].items():
                    y = yscale*vals[1]*xfunc(vals[0])
                    if y > 0.0:
                        y += yoffset
                        align = 'bottom'
                    else:
                        y -= yoffset
                        align = 'top'
                    subp[1].text(vals[0]+xcal_off, y, str(state), rotation=90,
                                 ha='center', va=align,
                                 fontsize='small', color=c_H)

                plt.subplots_adjust(hspace=0)
                subp[0].set_ylim(Yrang[rng+'_ram'][0], Yrang[rng+'_ram'][1])
                subp[1].set_ylim(Yrang[rng+'_roa'][0], Yrang[rng+'_roa'][1])
                subp[1].set_xlabel('Wavenumbers / cm$^{-1}$', fontsize=12)
                # format the y number

                subp[0].yaxis.set_ticks(Yrang[rng+'_ram'][3])
                subp[1].yaxis.set_ticks(Yrang[rng+'_roa'][3])
                subp[0].legend(loc='upper left', ncol=3)
                # subp[0].xaxis.set_ticks([])
                subp[0].tick_params(axis='x', direction='in', which='both',
                                    top=True, labeltop=True)
                subp[0].grid(axis='x', linestyle='--', linewidth='0.2')
                subp[1].grid(axis='x', linestyle='--', linewidth='0.2')
                subp[0].set_ylabel(
                    fmt_I_RS.format(int(log10(Yrang[rng+'_ram'][2]))))
                subp[1].set_ylabel(
                    fmt_I_ROA.format(int(log10(Yrang[rng+'_roa'][2]))))

                # plt.savefig(imgfile_assign.format(rng))
            plt.show()
