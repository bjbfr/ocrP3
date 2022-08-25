from ast import Return
from IPython.display import display

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns

import math

class pd_context:
    def __init__(self,options):
        self.shadows = {key:pd.get_option(key) for key in options.keys()}
        self.options = options
        
    @staticmethod
    def apply_options(options):
        _ = list(map(lambda t: pd.set_option(t[0], t[1]),options.items() )  )      
        
    def __enter__(self):
        pd_context.apply_options(self.options)
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pd_context.apply_options(self.shadows)

#test
#with pd_context({'display.max_columns':2}):
#    print(pd.get_option('display.max_columns'))
#print(pd.get_option('display.max_columns'))

def countplot(df,x=None,y=None,rot=45,hue=None,order=None,hue_order=None,stat=lambda x: x.value_counts(),figsize=(30, 12)):
    
    if stat is not None:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1,3,figure=fig)
        ax1 = fig.add_subplot(gs[:,:2])
        ax2 = fig.add_subplot(gs[:,-1])
    else:
        fig,ax1 = plt.subplots(figsize=figsize)
    
    c = x if x is not None else y
    #seaborn plot
    #filter order and hue order
    if order is not None:
        values = df[c].unique()
        order = list(filter(lambda x: x in values, order))
    if hue_order is not None:
        values = df[hue].unique()
        hue_order = list(filter(lambda x: x in values, hue_order))

    _ = sns.countplot(data=df,x=x,y=y,hue=hue,order=order,hue_order=hue_order,ax=ax1)

    # set label
    df_ = df[df[c].notna()]
    nb_values = len(df_[c].unique())
    label = ax1.get_xlabel() + f' ( {nb_values} values)'
    ax1.set_xlabel(label)
    
    if rot is not None:
        _ = ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rot, ha='right')
    
    if stat is not None:
        display_table(ax2,stat(df[x]))
    

def display_sample(df,max=10):
    sample_size = math.ceil(len(df)*0.15)
    N = min(sample_size,max)
    with pd_context({'display.max_columns':len(df.columns),'display.max_rows':N}):
        display(df.sample(sample_size).head(N))

def display_all(df):
    (n,m) =  len(df),len(df.columns)
    with pd_context({'display.max_columns':m,'display.max_rows':n,'display.max_colwidth':None}):
        display(df)
        print(f'{(n,m)}')

class spd:
    @staticmethod    
    def select(x,cols=None,rows=None):
        if x.ndim == 2:    
            return x.loc[
                    x.index   if rows is None else rows, 
                    x.columns if cols is None else cols
                ]    
        elif x.ndim == 1:
            rows = (rows if rows is not None else cols)
            return x.loc[x.index if rows is None else rows]

    @staticmethod
    def select_i(x,rows=None,cols=None):
        if x.ndim == 2:    
            return x.iloc[
                    slice(len(x.index))   if rows is None else rows, 
                    slice(len(x.columns)) if cols is None else cols
                ]    
        elif x.ndim == 1:
            rows = (rows if rows is not None else cols)
            return x.iloc[slice(len(x.index)) if rows is None else rows]

def clear_axis(ax):
    fs = [plt.Axes.set_xticks,plt.Axes.set_yticks,plt.Axes.set_xticklabels,plt.Axes.set_yticklabels]
    _ = list(map(lambda f: f(ax,()),fs ) ) 
    sns.despine(ax=ax,left=True,bottom=True)

def make_round(n):
    def r(x):
        if (x - math.floor(x)) < 1/(10**(n+1)):
            return  (f'{x:_.0f}').replace('_',' ')
        else: 
            return  (f'{x:_.{n}f}').replace('_',' ')
    return r

def choose_round(nb_decimal):
    if nb_decimal is None:
        return round_2
    else:
        return make_round(nb_decimal)

round_4 = make_round(4)
round_2 = make_round(2)
round_0 = make_round(0)
percent = lambda x: f'{x*100:.2f}%'

## A améliorer
# créer une table à partir de matplotlib.table.Table
# puis ajouter des cellules (add_cell) (ce qui permet d'affiner la taille des cellules)
# ajouter des fonctions de formatages des données
def display_table(ax,stat,nb_decimal=None):
    round = choose_round(nb_decimal)
    def to_str(x):
        if isinstance(x,tuple):
            return f"{round(x[0])} | {percent(x[1])}"
        else:
            return round(x)

    clear_axis(ax)
    #ax.axis('tight')
    #ax.axis('off')
    loc_ = 'center right'
    #loc_ = 'center'
    cellLoc_ = 'center'
    fontsize_ = 14
    t = None
    if isinstance(stat,pd.core.series.Series):
        cells = list(map(lambda x: [to_str(x)],stat.values))
        t = ax.table(cellText=cells,rowLabels=stat.index,colLabels=None,loc=loc_,cellLoc=cellLoc_,colWidths=[0.4])  #
    else: 
        t = ax.table(cellText=stat.values,colLabels=[stat.name],rowLabels=stat.index,loc=loc_,cellLoc=cellLoc_)
    #t.auto_set_font_size(False)
    #t.set_fontsize(fontsize_)
    t.scale(1,2)
    return t

def univariate_num_stat(serie,olimits=True,onas=True,ozeros=True,nb_decimal=None):
    """Compute basic statistics of quantitative variable"""
    round = choose_round(nb_decimal)
    stat = serie.describe()
    #compute various ratios
    N = len(serie)
    def count_ratio(cond,label):
        nb  = len(serie[cond])
        stat[label] = (nb,nb/N)
    # 1) zeros   
    if ozeros == True:
        count_ratio((serie == 0.0),'zeros')
    #2) nas
    if onas == True:
        count_ratio((serie.isna()),'nas')
    #3) iqr limits
    limits = None
    if olimits == True:
        iqr = stat['75%'] - stat['25%']
        limits = {f'{f}*IQR':(stat['25%'] - f*iqr,stat['75%'] + f*iqr) for f in [1.5,3,5]}
        def apply_count_ratio(key):
            count_ratio((serie < limits[key][0]),f'< Q1 - {key} ({round(limits[key][0])})')
            count_ratio((serie > limits[key][1]),f'> Q3 + {key} ({round(limits[key][1])})')
        _ = list(map(apply_count_ratio, limits.keys()))
    
    return (stat,limits)

def display_univariate_num(axes,serie,lines=[],limits=None,colors=None):
    with sns.axes_style('ticks',rc={'axes.grid':True}):
        #axes[0]
        ## check if too many bins to be displayed
        bin_edges = np.histogram_bin_edges(serie[serie.notna()],bins='auto')
        bins_ = 'auto'
        if len(bin_edges) > 1000:
            bins_ = 'sqrt'
        ######################################################
        _ = sns.histplot(x=serie,stat='count',kde=True,ax=axes[0],bins=bins_)
        axes[0].set_xlabel('')
        #axes[1]
        _ = sns.boxplot(x=serie,orient='h',ax=axes[1])
        axes[1].set_xlabel('')
        ylims = axes[1].get_ylim()
        for line in lines:
            (lmin,lmax) = limits[line]
            color_ = colors[line]
            low_limit  = Line2D([lmin,lmin],ylims,color=color_,ls='--')
            high_limit = Line2D([lmax,lmax],ylims,color=color_,ls='--')
            axes[1].add_line(low_limit)
            axes[1].add_line(high_limit)

def layout_wtable(N):
    figsize_ = (36,N*12)
    fig = plt.figure(figsize=figsize_)
    gs = GridSpec(N,5,figure=fig)
    axes = [ ((fig.add_subplot(gs[n:(n+1),:2]),fig.add_subplot(gs[n:(n+1),2:4])), fig.add_subplot(gs[n:(n+1),-1])) for n in range(N) ]
    return (fig,axes)

def univariate_num(serie,filters=[],customFilters=[],nb_decimal=None):
    fontsize_ = 14
    fontweight_ = "bold"
    _title_ = lambda p: f" {serie.name if serie.name is not None else ''} {p[0]} {p[1]}"
    colors ={'1.5*IQR':'green','3*IQR':'blue','5*IQR':'red'}

    filters.sort()
    filters.reverse()
    
    (fig,axes) = layout_wtable(len(filters)+len(customFilters)+1) 
    (stat,limits) = univariate_num_stat(serie,nb_decimal=nb_decimal)
    
    # affichage non-filtré
    display_univariate_num(axes[0][0],serie,['1.5*IQR','3*IQR','5*IQR'],limits,colors)
    # table stat
    _ = display_table(axes[0][1],stat,nb_decimal)
    (axes[0][1]).set_title(_title_(("Non-filtered",'')),fontsize=fontsize_,fontweight=fontweight_)
    #add legend
    legend_handles = [ Line2D([],[],color=c,label=l)  for (l,c) in colors.items()]
    (axes[0][1]).legend(handles=legend_handles,loc='upper center')

    # affichage filtré (filtres pré-définis)
    lines = filters.copy()
    for (i,filter) in enumerate(filters):
        (lmin,lmax) = limits[filter]
        filtered_serie = serie[(serie >= lmin ) & (serie <= lmax)]
        (stat_,_) = univariate_num_stat(filtered_serie,nb_decimal=nb_decimal,olimits=False,onas=False)
        lines = [f for f in lines if f != filter]
        display_univariate_num(axes[i+1][0],filtered_serie,lines,limits,colors)
        # table stat
        _ = display_table(axes[i+1][1],stat_,nb_decimal)
        (axes[i+1][1]).set_title(_title_(('Filtered',filter)),fontsize=fontsize_,fontweight=fontweight_)

    # affichage filtré (filtres personnalisés)
    for (i,f) in enumerate(customFilters):
        t = type(f)
        if t is tuple:
            (lmin,lmax) =  t
            filtered_serie = serie[(serie >= lmin ) & (serie <= lmax)]
            label = f'[{lmin},{lmax}]'
        elif t is float or t is int:
            filtered_serie = serie[serie != f]
            label = f'<> {f}'

        (stat_,_) = univariate_num_stat(filtered_serie,nb_decimal=nb_decimal,olimits=False,onas=False)    
        display_univariate_num(axes[i+1][0],filtered_serie)
        # table stat
        _ = display_table(axes[i+1][1],stat_,nb_decimal)
        (axes[i+1][1]).set_title(_title_(('Filtered',label)),fontsize=fontsize_,fontweight=fontweight_)

def univariate_num_partition_filter(df,part_col,filter_):
    if filter_ not in ['1.5*IQR','3*IQR','5*IQR']:
        return
    (_,limits) = univariate_num_stat(df[part_col],olimits=True,onas=False,ozeros=False)
    (lmin,lmax) = limits[filter_]
    res_col = f'{part_col}_{filter_}_part'
    df.loc[df[df[part_col] < lmin].index,res_col]  = -1
    df.loc[df[df[part_col] > lmax].index,res_col]  =  1
    df.loc[df[df[res_col].isna() ].index,res_col]  =  0
    return res_col
    