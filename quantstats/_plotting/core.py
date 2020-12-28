#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
#
# Copyright 2019 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as _plt
try:
    _plt.rcParams["font.family"] = "Arial"
except Exception:
    pass

import matplotlib.dates as _mdates
from matplotlib.ticker import (
    FormatStrFormatter as _FormatStrFormatter,
    FuncFormatter as _FuncFormatter
)

import pandas as _pd
import numpy as _np
import seaborn as _sns
from .. import stats as _stats
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as ep
import datetime
import math

_sns.set(font_scale=1.1, rc={
    'figure.figsize': (10, 6),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666'
})

_FLATUI_COLORS = ["#fedd78", "#348dc1", "#af4b64",
                  "#4fa487", "#9b59b6", "#808080"]
_GRAYSCALE_COLORS = ['silver', '#222222', 'gray'] * 3


def _get_colors(grayscale):
    colors = _FLATUI_COLORS
    ls = '-'
    alpha = .8
    if grayscale:
        colors = _GRAYSCALE_COLORS
        ls = '-'
        alpha = 0.5
    return colors, ls, alpha


def plot_returns_bars(returns, benchmark=None,
                      returns_label="Strategy",
                      hline=None, hlw=None, hlcolor="red", hllabel="",
                      resample="A", title="Returns", match_volatility=False,
                      log_scale=False, figsize=(10, 6),
                      grayscale=False, fontname='Arial', ylabel=True,
                      subtitle=True, savefig=None, show=True, fig_type="plotly"):

    if match_volatility and benchmark is None:
        raise ValueError('match_volatility requires passing of '
                         'benchmark.')
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    colors, _, _ = _get_colors(grayscale)
    df = _pd.DataFrame(index=returns.index, data={returns_label: returns})
    if isinstance(benchmark, _pd.Series):
        df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
        df = df[['Benchmark', returns_label]]
    
    df = df.dropna()
    if resample is not None:
        df = df.resample(resample).apply(
            _stats.comp).resample(resample).last()
    # ---------------
   
    # import pdb; pdb.set_trace()
    
     
    if fig_type == "plotly":
        df["color"] = 'rgb(255, 255, 255)'
        df.loc[df[returns_label] < 0.0, 'color'] = 'rgb(0, 255, 0)'
        df.loc[df[returns_label] > 0.0, 'color'] = 'rgb(255, 0, 0)'
        # df['width'] = 5
        df.index = df.index.strftime('%Y年')
        if len(df) < 4:
            bargap = 0.9
        else:
            bargap = 0.5
        fig_dict = {
            "data": [{
                'type': 'bar',
                'x': df.index,
                'y': df[returns_label],
                'marker': dict(color=df['color'].tolist())
                # 'width': df['width'].tolist()
            }],
            "layout": {
                "title": title,
                "bargap": bargap
            }
        }

        fig = go.Figure(fig_dict)
        fig.update_layout(margin=dict(l=10, r=10))
        fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
        fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y年", yaxis_tickformat=".2%")
        fig.update_layout(title_text=title, title_x=0.5)
        fig.update_xaxes(tickangle=45, title_font={"size": 20}, title_standoff=0)
           
        # fig.add_hline(y=0.1)
        # fig.add_vline(x=0.1)
        if hline:
            if grayscale:
                hlcolor = 'black'
            fig.add_hline(y=hline, line_dash="dot", line_width=3, line_color=hlcolor,
              annotation_text=hllabel, 
              annotation_position="top right") 

        # if len(df) < 5:
        #     fig.update_layout(xaxis_range=[datetime.datetime(2020, 1, 5), datetime.datetime(2020, 1, 6)])
      
     
        return fig
        
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # use a more precise date string for the x axis locations in the toolbar
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            df.index.date[:1][0].strftime('%Y'),
            df.index.date[-1:][0].strftime('%Y')
        ), fontsize=12, color='gray')

    if benchmark is None:
        colors = colors[1:]
    df.plot(kind='bar', ax=ax, color=colors)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_xticklabels(df.index.year)
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
    years = sorted(list(set(df.index.year)))
    if len(years) > 10:
        mod = int(len(years)/10)
        _plt.xticks(_np.arange(len(years)), [
            str(year) if not i % mod else '' for i, year in enumerate(years)])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    if hline:
        if grayscale:
            hlcolor = 'gray'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    if isinstance(benchmark, _pd.Series) or hline:
        ax.legend(fontsize=12)

    _plt.yscale("symlog" if log_scale else "linear")

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel("Returns", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_timeseries(returns, benchmark=None,
                    title="Returns", compound=False, cumulative=True,
                    fill=False, returns_label="Strategy",
                    hline=None, hlw=None, hlcolor="red", hllabel="",
                    percent=True, match_volatility=False, log_scale=False,
                    resample=None, lw=1.5, figsize=(10, 6), ylabel="",
                    grayscale=False, fontname="Arial",
                    subtitle=True, savefig=None, show=True, fig_type="plotly", titles=None):

    colors, ls, alpha = _get_colors(grayscale)

    if isinstance(returns, list) == False:
        returns = [returns]
    [ret.fillna(0, inplace=True) for ret in returns]
    if isinstance(benchmark, _pd.Series):
        benchmark.fillna(0, inplace=True)

    if match_volatility and benchmark is None:
        raise ValueError('match_volatility requires passing of '
                         'benchmark.')
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.std()
        returns = [(ret/ ret.std()) * bmark_vol for ret in returns]

    # ---------------
    if compound is True:
        if cumulative:
            returns = [_stats.compsum(ret) for ret in returns]
            if isinstance(benchmark, _pd.Series):
                benchmark = _stats.compsum(benchmark)
        else:
            returns = returns.cumsum()
            if isinstance(benchmark, _pd.Series):
                benchmark = benchmark.cumsum()

    if resample:
        # returns = returns.resample(resample)
        returns = [ret.resample(resample) for ret in returns]
        returns = [ret.last() if compound is True else ret.sum() for ret in returns]
        if isinstance(benchmark, _pd.Series):
            benchmark = benchmark.resample(resample)
            benchmark = benchmark.last(
            ) if compound is True else benchmark.sum()
    # ---------------
    
    # using pltoly
    if fig_type=="plotly":
        filled = None
        if fill:
            filled = "tozeroy"

        data = []
        i = 0
        for ret in returns:
            title = ""
            if titles and len(titles) > i:
                title = titles[i]

            i = i + 1
            data.append({
                "type": 'scatter',
                'x': ret.index,
                'y': ret.values,
                'fill': filled,
                'name': title
            })
        if isinstance(benchmark, _pd.Series):
            data.append({
                "type": "scatter",
                'x': benchmark.index,
                'y': benchmark.values,
                'fill': filled,
                "name": titles[i]
            })
        fig_dict = {
            "data": data,
            "layout": {
                "title": title
            }
        }

        fig = go.Figure(fig_dict)
        # fig.update_layout(margin=dict(l=10, r=1))
        # fig.update_layout(legend=dict(
        #         orientation="h",
        #         yanchor="bottom",
        #         y=1.02,
        #         xanchor="right",
        #         x=1
        #     ))
        fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y年%m月%d日", yaxis_tickformat=".2%")
        # fig.update_layout(title_text=title, title_x=0.5, title_text__title="hello")
        fig.update_xaxes(tickangle=45, title_font={"size": 20}, title_standoff=25)
        if hline:
            if grayscale:
                hlcolor = 'black'
            fig.add_hline(y=hline, line_width=3, line_dash="dash", line_color=hlcolor)
        # fig.update_layout(paper_bgcolor="rgba(11, 11, 11, 11)")
        return fig

        
        





    # using matplotlib
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                  " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    if isinstance(benchmark, _pd.Series):
        ax.plot(benchmark, lw=lw, ls=ls, label="Benchmark", color=colors[0])

    alpha = .25 if grayscale else 1
    ax.plot(returns, lw=lw, label=returns_label, color=colors[1], alpha=alpha)

    if fill:
        ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=.25)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="-", lw=1,
               color='gray', zorder=1)
    ax.axhline(0, ls="--", lw=1,
               color='white' if grayscale else 'black', zorder=2)

    if isinstance(benchmark, _pd.Series) or hline:
        ax.legend(fontsize=12)

    _plt.yscale("symlog" if log_scale else "linear")

    if percent:
        ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
        # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        #     lambda x, loc: "{:,}%".format(int(x*100))))

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel(ylabel, fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_histogram(returns, resample="M", bins=100,
                   fontname='Arial', grayscale=False,
                   title="Returns", kde=True, figsize=(10, 6),
                   ylabel=True, subtitle=True, compounded=True,
                   savefig=None, show=True, fig_type="plotly"):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['silver', 'gray', 'black']

    apply_fnc = _stats.comp if compounded else _np.sum
    returns = returns.fillna(0).resample(resample).apply(
        apply_fnc).resample(resample).last()


    if fig_type=="plotly":
        # fig = ff.create_distplot([returns.values], [title], bin_size=0.01)
        fig = ep.histogram(x=returns, nbins=bins)
        
        fig.add_vline(x=0.0, line_width=1, line_color="black")
        fig.add_vline(x=returns.mean(), line_color="red", line_dash="dot")
        fig.update_layout(xaxis_tickformat=".2%")
        fig.update_layout(title_text=title, title_x=0.5, yaxis_title="数量", xaxis_title="收益率")
        fig.update_traces(hovertemplate='收益率:%{x}<br>月份数:%{y}')
        fig.update_layout(margin=dict(l=10, r=10))
        return fig

        
        # fig_dict = {
        #     "data": [{
        #         'type': 'scatter',
        #         'x': returns.index,
        #         'y': returns.values
        #     }],
        #     "layout": {
        #         "title": title
        #     }
        # }

        # fig = go.Figure(fig_dict)
        # fig.update_layout(margin=dict(l=10, r=10))
        # fig.update_layout(legend=dict(
        #         orientation="h",
        #         yanchor="bottom",
        #         y=1.02,
        #         xanchor="right",
        #         x=1
        #     ))
        # fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y年%m月", yaxis_tickformat=".2%")
        # fig.update_layout(title_text=title, title_x=0.5)
        # fig.update_xaxes(tickangle=45, title_font={"size": 20}, title_standoff=25)
        # if hline:
        #     if grayscale:
        #         hlcolor = 'black'
        #     fig.add_vline(x=hline, line_width=3, line_dash="dash", line_color=hlcolor)
        # return fig

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%Y'),
            returns.index.date[-1:][0].strftime('%Y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.axvline(returns.mean(), ls="--", lw=1.5,
               color=colors[2], zorder=2, label="Average")

    _sns.distplot(returns, bins=bins,
                  axlabel="", color=colors[0], hist_kws=dict(alpha=1),
                  kde=kde,
                  # , label="Kernel Estimate"
                  kde_kws=dict(color='black', alpha=.7),
                  ax=ax)

    ax.xaxis.set_major_formatter(_plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    ax.axvline(0, lw=1, color="#000000", zorder=2)

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel("Occurrences", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.legend(fontsize=12)

    # fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_rolling_stats(returns, benchmark=None, title="",
                       returns_label="Strategy",
                       hline=None, hlw=None, hlcolor="red", hllabel="",
                       lw=1.5, figsize=(10, 6), ylabel="",
                       grayscale=False, fontname="Arial", subtitle=True,
                       savefig=None, show=True):

    colors, _, _ = _get_colors(grayscale)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    df = _pd.DataFrame(index=returns.index, data={returns_label: returns})
    if isinstance(benchmark, _pd.Series):
        df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
        df = df[['Benchmark', returns_label]].dropna()
        ax.plot(df['Benchmark'], lw=lw, label="Benchmark",
                color=colors[0], alpha=.8)

    ax.plot(df[returns_label].dropna(), lw=lw,
            label=returns_label, color=colors[1])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')\
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            df.index.date[:1][0].strftime('%e %b \'%y'),
            df.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    if ylabel:
        ax.set_ylabel(ylabel, fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(_FormatStrFormatter('%.2f'))

    ax.legend(fontsize=12)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)
    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_rolling_beta(returns, benchmark,
                      window1=126, window1_label="",
                      window2=None, window2_label="",
                      title="", hlcolor="red", figsize=(10, 6),
                      grayscale=False, fontname="Arial", lw=1.5,
                      ylabel=True, subtitle=True, savefig=None, show=True):

    colors, _, _ = _get_colors(grayscale)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    beta = _stats.rolling_greeks(returns, benchmark, window1)['beta']
    ax.plot(beta, lw=lw, label=window1_label, color=colors[1])

    if window2:
        ax.plot(_stats.rolling_greeks(returns, benchmark, window2)['beta'],
                lw=lw, label=window2_label, color="gray", alpha=0.8)
    mmin = min([-100, int(beta.min()*100)])
    mmax = max([100, int(beta.max()*100)])
    step = 50 if (mmax-mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    hlcolor = 'black' if grayscale else hlcolor
    ax.axhline(beta.mean(), ls="--", lw=1.5,
               color=hlcolor, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    if ylabel:
        ax.set_ylabel("Beta", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.legend(fontsize=12)
    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_longest_drawdowns(returns, periods=5, lw=1.5,
                           fontname='Arial', grayscale=False,
                           log_scale=False, figsize=(10, 6), ylabel=True,
                           subtitle=True, compounded=True,
                           savefig=None, show=True, fig_type="plotly"):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['#000000'] * 3

    series = _stats.compsum(returns) if compounded else returns.cumsum()
    dd = _stats.to_drawdown_series(returns.fillna(0))
    dddf = _stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(
        by='days', ascending=False, kind='mergesort')[:periods]

    if fig_type == "plotly":
        fig_dict = {
            "data": [{
                'type': 'scatter',
                'x': series.index,
                'y': series.values
            }]
            # "layout": {
            #     "title": title
            # }
        }

        fig = go.Figure(fig_dict)
        fig.update_layout(margin=dict(l=10, r=10))
        fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))

        title = f"前{len(longest_dd)}大回撤区间分布图"
        for _, row in longest_dd.iterrows():
            # fig.add_vrect(x0="2018-09-24", x1="2018-12-18", row="all", col=1,
            #   annotation_text="decline", annotation_position="top left",
            #   fillcolor="green", opacity=0.25, line_width=0)
            # fig.add_vrect(x0=row['start'], x1=row['end'], row="all", col=1,
            text = f"{row['start']}~{row['end']}"
            fig.add_vrect(x0=row['start'], x1=row['end'], 
            #   annotation_text=text, annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)
        # ax.axvspan(*_mdates.datestr2num([str(row['start']), str(row['end'])]),
        #            color=highlight, alpha=.1)
        fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y年%m月", yaxis_tickformat=".2%")
        fig.update_layout(title_text=title, title_x=0.5)
        fig.update_xaxes(tickangle=45, title_font={"size": 20}, title_standoff=25)
     
        return fig


    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle("Top %.0f Drawdown Periods\n" %
                 periods, y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")
    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    series = _stats.compsum(returns) if compounded else returns.cumsum()
    ax.plot(series, lw=lw, label="Backtest", color=colors[0])

    highlight = 'black' if grayscale else 'red'
    for _, row in longest_dd.iterrows():
        ax.axvspan(*_mdates.datestr2num([str(row['start']), str(row['end'])]),
                   color=highlight, alpha=.1)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)
    _plt.yscale("symlog" if log_scale else "linear")
    if ylabel:
        ax.set_ylabel("Cumulative Returns", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
    # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
    #     lambda x, loc: "{:,}%".format(int(x*100))))

    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_distribution(returns, figsize=(10, 6),
                      fontname='Arial', grayscale=False, ylabel=True,
                      subtitle=True, compounded=True,
                      savefig=None, show=True, fig_type="plotly"):

    colors = _FLATUI_COLORS
    if grayscale:
        colors = ['#f9f9f9', '#dddddd', '#bbbbbb', '#999999', '#808080']
    # colors, ls, alpha = _get_colors(grayscale)

    port = _pd.DataFrame(returns.fillna(0))
    port.columns = ['Daily']

    apply_fnc = _stats.comp if compounded else _np.sum

    port['Weekly'] = port['Daily'].resample(
        'W-MON').apply(apply_fnc).resample('W-MON').last()
    port['Weekly'].ffill(inplace=True)

    port['Monthly'] = port['Daily'].resample(
        'M').apply(apply_fnc).resample('M').last()
    port['Monthly'].ffill(inplace=True)

    port['Quarterly'] = port['Daily'].resample(
        'Q').apply(apply_fnc).resample('Q').last()
    port['Quarterly'].ffill(inplace=True)

    port['Yearly'] = port['Daily'].resample(
        'A').apply(apply_fnc).resample('A').last()
    port['Yearly'].ffill(inplace=True)

    port = port.rename(columns={"Daily": "日" , "Weekly": "周", "Monthly": "月", "Quarterly": "季", "Yearly": "年"})

  
    if fig_type == "plotly":
        fig = ep.box(port, y=port.columns)
        fig.update_layout(yaxis_tickformat=".2%")
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="收益率")
        fig.update_layout(margin=dict(l=10, r=10))
        # fig.update_layout(paper_bgcolor="rgba(11, 11, 11, 11)")
        return fig
    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle("Return Quantiles\n", y=.99,
                 fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    _sns.boxplot(data=port, ax=ax, palette=tuple(colors[:5]))

    ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    if ylabel:
        ax.set_ylabel('Rerurns', fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def plot_table(tbl, columns=None, title="", title_loc="left",
               header=True,
               colWidths=None,
               rowLoc='right',
               colLoc='right',
               colLabels=None,
               edges='horizontal',
               orient='horizontal',
               figsize=(5.5, 6),
               savefig=None,
               show=False):

    if columns is not None:
        try:
            tbl.columns = columns
        except Exception:
            pass

    fig = _plt.figure(figsize=figsize)
    ax = _plt.subplot(111, frame_on=False)

    if title != "":
        ax.set_title(title, fontweight="bold",
                     fontsize=14, color="black", loc=title_loc)

    the_table = ax.table(cellText=tbl.values,
                         colWidths=colWidths,
                         rowLoc=rowLoc,
                         colLoc=colLoc,
                         edges=edges,
                         colLabels=(tbl.columns if header else colLabels),
                         loc='center',
                         zorder=2
                         )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_height(0.08)
        cell.set_text_props(color='black')
        cell.set_edgecolor('#dddddd')
        if row == 0 and header:
            cell.set_edgecolor('black')
            cell.set_facecolor('black')
            cell.set_linewidth(2)
            cell.set_text_props(weight='bold', color='black')
        elif col == 0 and "vertical" in orient:
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(1)
            cell.set_text_props(weight='bold', color='black')
        elif row > 1:
            cell.set_linewidth(1)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    # _plt.close()

    if not show:
        return fig

    return None


def format_cur_axis(x, _):
    if x >= 1e12:
        res = '$%1.1fT' % (x * 1e-12)
        return res.replace('.0T', 'T')
    if x >= 1e9:
        res = '$%1.1fB' % (x * 1e-9)
        return res.replace('.0B', 'B')
    if x >= 1e6:
        res = '$%1.1fM' % (x * 1e-6)
        return res.replace('.0M', 'M')
    if x >= 1e3:
        res = '$%1.0fK' % (x * 1e-3)
        return res.replace('.0K', 'K')
    res = '$%1.0f' % x
    return res.replace('.0', '')


def format_pct_axis(x, _):
    x *= 100  # lambda x, loc: "{:,}%".format(int(x * 100))
    if x >= 1e12:
        res = '%1.1fT%%' % (x * 1e-12)
        return res.replace('.0T%', 'T%')
    if x >= 1e9:
        res = '%1.1fB%%' % (x * 1e-9)
        return res.replace('.0B%', 'B%')
    if x >= 1e6:
        res = '%1.1fM%%' % (x * 1e-6)
        return res.replace('.0M%', 'M%')
    if x >= 1e3:
        res = '%1.1fK%%' % (x * 1e-3)
        return res.replace('.0K%', 'K%')
    res = '%1.0f%%' % x
    return res.replace('.0%', '%')
