import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io
import os
import plotly as py
import plotly.graph_objects as go
import ipywidgets as widget
import plotly.subplots as sb
import numpy as np
import scipy.stats as stats
import plotly.io as pio
# import docx
import tempfile
# from docx.shared import Inches
import streamlit as st


py.offline.init_notebook_mode()


plotly_colors = [
    'rgb(31, 119, 180)',  # Blue
    'rgb(255, 127, 14)',  # Orange
    'rgb(44, 160, 44)',   # Green
    'rgb(214, 39, 40)',   # Red
    'rgb(148, 103, 189)', # Purple
    'rgb(140, 86, 75)',   # Brown
    'rgb(227, 119, 194)', # Pink
    'rgb(127, 127, 127)', # Gray
    'rgb(188, 189, 34)',  # Olive
    'rgb(23, 190, 207)',  # Teal
    'rgb(140, 140, 140)', # Dark Gray
    'rgb(255, 187, 120)', # Light Orange
    'rgb(44, 50, 180)',   # Navy Blue
    'rgb(255, 152, 150)', # Light Red
    'rgb(214, 39, 40)'    # Red (same as above for emphasis)
]

# Making segments
def make_segments(df, total_segments = 10, num_dp = 12000):
    num_dp = int(df.shape[0])// total_segments
    df_seg = np.empty((total_segments,num_dp))  # numpy array of 10 seg of 12000 dp each 

    for i in range(total_segments):
        start_index = i * num_dp
        end_index = start_index + num_dp
        df_seg[i] = df[start_index:end_index]
        
    return df_seg  # return np.array of shape (10,12000)




# Plotting Data
def plot_data(df,
                num_segments = 1,
                xlabel='x', 
                ylabel='y', 
                title='Graph', 
                seg_num=1,
                sampling_freq = 12000,):
    

    rows = 5
    cols = 2
    if num_segments == 1:
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x = np.arange(0, df[seg_num-1].shape[-1]/(sampling_freq -1), 1/sampling_freq),
                                 y=df[seg_num - 1],
                                 mode='lines',
                                 line=dict(color='blue'),
                                 name=f'{title} |  Segment {seg_num}',
                                 showlegend=False,
                                 
                                 ))
        
        fig.update_xaxes(title_text="Time(s)",
                         showgrid=True, 
                         gridcolor='lightgray',)
        
        fig.update_yaxes(title_text="Amplitude" ,
                         showgrid=True, 
                         gridcolor='lightgray',)
        
    else:
        fig = sb.make_subplots(rows=rows, 
                               cols=cols, 
                               subplot_titles=[f'Segement {i+1}' for i in range(num_segments)],
                               vertical_spacing=0.07,
                               row_heights=[0.20, 0.20, 0.20, 0.20, 0.20])
        

        for i in range(num_segments):
            row = (i // cols) + 1
            col = (i % cols) + 1

            fig.add_trace(go.Scatter(
                x = np.arange(0, 1, 1/sampling_freq),
                y=df[i], mode='lines',
                line=dict(color='blue'),
                name=f'Segment {i+1}',
                showlegend=False), 
                row=row, col=col)

            fig.update_xaxes(title_text=xlabel, row=row, col=col)
            fig.update_yaxes(title_text=ylabel, row=row, col=col)
        

    fig.update_layout(title = {
                        'text': title + f' | (Segment {seg_num})',
                        'font': {
                        'family': 'Arial Bold',
                         'size': 25
                         }}, 
                     title_x = 0.40,
                     height=1500 if num_segments != 1 else 500,
                     width=900,
                     plot_bgcolor='white',
                     showlegend=False,
                     xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
                     yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
                     margin=dict(l=50, r=50, t=80, b=50),
                     
                     )
    
    
    fig.update_traces(hovertemplate='Time: %{x}s<br>Amp: %{y}') 
    return fig
    
    
#--------------------------------------------------------------------------------------------------------------#   
    
    

import scipy.fft as sp

# Applying FFT
def convert_df(df_seg, total_segments = 10):  # df_seg ----> segmented data
    
    df_freqseg = np.empty((df_seg.shape)) # np array to store in frequency domain

    for i in range(total_segments):
        df_freqseg[i] = np.abs(scipy.fft.fft(df_seg[i]))
        
    return df_freqseg


#---------------------------------------------------------------------------------------------------------------#

# Plotting FFt

def plot_fft(x_axis, y_axis,num_segments = 1, 
             xlabel='x', ylabel='y', 
             title='Graph', seg_num = 1,
             save_img=False, 
             save_path=''):
    
    if num_segments == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = x_axis,y = y_axis[seg_num - 1],
                                 mode='lines', line=dict(color='blue'),
                                 name=f'Segment {seg_num}',showlegend=False))
        
        fig.update_xaxes(title_text='Frequency (Hz)',
                         showgrid=True, 
                         gridcolor='lightgray', 
                         )
        
        fig.update_yaxes(title_text='Amplitude',
                        showgrid=True, 
                         gridcolor='lightgray', 
                        )
    else:    
        rows=(num_segments+1)//2
        fig = sb.make_subplots(rows=rows, cols=2, 
                               subplot_titles=[f'Segment {i+1}' for i in range(num_segments)],
                               row_heights=list(np.full((rows,), 1/rows)),
                               column_widths=[0.5,0.5],
                               vertical_spacing=0.1)

        for i in range(num_segments):
            fig.add_trace(go.Scatter(
                x=x_axis, y=y_axis[i], mode='lines', 
                name=f'Segment {i+1}',
                line=dict(color='blue')), row=(i // 2) + 1, col=(i % 2) + 1)
    
        
            fig.update_xaxes(title_text=xlabel, 
                             row=(i // 2) + 1, col=(i % 2) + 1,
                             showline=True, 
                             linewidth=1, 
                             linecolor='black')
            
            fig.update_yaxes(title_text=ylabel, 
                             row=(i // 2) + 1, col=(i % 2) + 1,
                             showline=True, 
                             linewidth=1, 
                             linecolor='black')
        
    fig.update_traces(hovertemplate='freq: %{x}<br><br>amp: %{y}')
    fig.update_layout(title={
        'text': title + f' | (Segment {seg_num})',
        'font': {
            'family': 'Arial Bold',
            'size': 18
        }},title_x = 0.4,
        hovermode='closest', 
        plot_bgcolor='white',
        showlegend=False,
        width=950,
        xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        margin=dict(l=50, r=50, t=80, b=50))
    
    fig.update_layout(height= 450 if num_segments == 1 else (rows * 300 if rows > 2 else (rows) * 450 ))
    
    if save_path and save_img:
        full_path = f"{save_path}/{title}.png"
        pio.write_image(fig, full_path)  # Save the image
    
    return fig
                      
        
def filter_freq(df_freqseg, amp = 0.2, file_name=''):
    if 'or' in file_name.lower():
        amp = 0.04

    fdf_freqseg = np.copy(df_freqseg)
    num = df_freqseg.shape[0]

    for i in range(num):
        max_amplitude = np.max(fdf_freqseg[i])
        threshold = amp * max_amplitude
        fdf_freqseg[i][fdf_freqseg[i] < threshold] = 0
    return fdf_freqseg

# Extracting Features

def extract_features(fdf_freqseg):
  
    df_features = {
        'mean': np.mean(fdf_freqseg, axis=1),
        'std': np.std(fdf_freqseg, axis=1),
        'variance': np.var(fdf_freqseg, axis=1),
        'max' : np.max(np.abs(fdf_freqseg), axis = 1),
        'rms': np.sqrt(np.mean(np.square(fdf_freqseg),axis=1)),
        'skewness': stats.skew(fdf_freqseg, axis=1),
        'kurtosis': stats.kurtosis(fdf_freqseg, axis=1),
        'crest_factor': np.max(fdf_freqseg, axis = 1)/ np.sqrt(np.mean(fdf_freqseg**2, axis=1)),
        'margin_factor' : np.max(fdf_freqseg, axis = 1)/np.var(fdf_freqseg, axis=1),
        'shape_factor' : (np.sqrt(np.mean(np.square(fdf_freqseg),axis=1))/np.mean(np.abs(fdf_freqseg), axis=1)),
        'impulse_factor' : np.max(np.abs(fdf_freqseg), axis = 1)/np.mean(np.abs(fdf_freqseg), axis=1),
    }
    
    return df_features


#--------------------------------------------------------------------------------------------------------------------
#------------------------------------------ PLOTTING  Features -------------------------------------------------------

def plot_features(df_features, title = "Features", keys=['Mean', 'STD', 'Skewness','Kurtosis'], 
             save_img=False, 
             save_path='', 
             save_doc=False,
             doc_name = 'first.docx'):
    
    """
Description:

This function takes a DataFrame (df_features) as input and generates multiple subplots for different statistical features of the data. The subplots include line plots for each segment in the DataFrame, representing the specified statistical feature. The function provides options to save the plot as an image and as a Word document, with customizable save path and document name.

Parameters:

df_features (DataFrame): The input DataFrame containing the features to be plotted.
title (str, optional): The title of the plot. Default is "Features".
keys (list, optional): The list of statistical features to be plotted. Default includes 'Mean', 'STD', 'Skewness', and 'Kurtosis'.
save_img (bool, optional): Whether to save the plot as an image. Default is False.
save_path (str, optional): The directory path where the plot image or document will be saved. Default is an empty string.
save_doc (bool, optional): Whether to save the plot as a Word document. Default is False.
doc_name (str, optional): The name of the Word document if save_doc is True. Default is 'first.docx'.
Returns: None

Raises:

ValueError if the length of keys is greater than 4.
    """
    
    titles = []

    for key in keys:
        titles.append(key)
        titles.append(key + '(sorted)')
    
    
    df_keys = list(df_features.keys())
    if len(df_keys) == 3:
        colors = ['red', 'blue', 'green']
    else:
        colors = plotly_colors[:len(df_keys)]
    rows = len(keys) if len(keys)<= 3 else 4
    cols = 2

    fig = sb.make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.15, horizontal_spacing=0.10)
    
    for i in range(rows):
        y_min = 0
        y_max = 0
         
        for j in range(cols):
            
            for color, segment in zip(colors,df_keys):
                y_data = df_features[segment][keys[i].lower()]
                if j == 1:
                    y_data = np.sort(y_data)
                    show_legend = False
                else:
                    if i == 0 and j==0:
                        show_legend = True
                    else:
                         show_legend = False
                fig.add_trace(go.Scatter(
                    # x= np.arange(1,11),
                     y=y_data, 
                    mode='lines+markers', name=segment, line=dict(color=color), showlegend=show_legend),
                    row=i + 1, col=j + 1)
                
                y_min = min(min(y_data), y_min)
                y_max = max(max(y_data), y_max)
                
 
                fig.update_xaxes(title_text='Segment', row=i + 1, col=j + 1,
                                 showgrid=True, 
                                 gridcolor='lightgray',
                                 showline=True, 
                                 linewidth=1, 
                                 linecolor='black',
                                #  mirror=True,
                            # tickvals = [2,4,6,8,10]
                            )
                
                fig.update_yaxes(title_text='Values', row=i + 1, col=j + 1,
                                 showgrid=True, 
                                 gridcolor='lightgray',
                                 showline=True, 
                                 linewidth=1, 
                                 linecolor='black',
                                 mirror=True,
                               
#                                tickvals = [ 
#                                    round(i ,3) if (y_min < 0 or y_max < 1)
#                                    else round(i,0)
#                                    for i in list(np.linspace(y_min,y_max ,num= 6))],
                             
                              title_standoff=0
                            )
                             
#                                 tickvals=[round(y_min + (i * interval), 3) if y_min < 0 
#                                           else round(y_min + (i * interval), 0)for i in range(5)],                         



    fig.update_layout(
        title={
        'text': title,
        'font': {
            'family': 'Arial Bold',
            'size': 18
        }},title_x = 0.42,
        hovermode='closest',
        plot_bgcolor='white',
        # xaxis=dict(title='Segment',showline=True, linewidth=1, linecolor='black', mirror=True),
        # yaxis=dict(rangemode='tozero', title='Value',showline=True, linewidth=1, linecolor='black', mirror=True),
        height= rows * 400,
        width= 950,
        margin=dict(l=50, r=50, t=70, b=50)
    )
    fig.update_traces(hovertemplate='Seg: %{x}<br>Value: %{y}')  # Display y-value on hover
    
    
    title_ = f"{title}"+ "_".join(keys)
    
# -------->>> SAVING IT AS A  Image
    if save_path and save_img:
        full_path = f"{save_path}/{title_}.png"
        pio.write_image(fig, full_path)  # Save the image   
        
# -------->>> SAVING IT AS A DOCUMENT

    # if save_doc and save_path:
    #     doc_path = os.path.join(save_path, doc_name)
    #     temp_doc = None
    #     if doc_path and os.path.exists(doc_path):
    #         document = docx.Document(doc_path)
    #         document.save(os.path.join(save_path, "temp.docx"))
    #         temp_doc = docx.Document(os.path.join(save_path, "temp.docx"))
    #         document = docx.Document()
    #     else:
    #         document = docx.Document()
            
            
     # Create a new paragraph with the title
        # document.add_paragraph(f'{title_}') 
    
        # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        #     image_path = tmpfile.name
        #     pio.write_image(fig, image_path)
        #     document.add_picture(image_path, width=Inches(4))
     
        # if temp_doc != None:
        #     for element in temp_doc.element.body:
        #         document.element.body.append(element)
    
   
        #     os.remove(os.path.join(save_path, "temp.docx"))

        # if doc_path is None:
        #     document.save(f"{doc_name}.docx")
        # else:
        #     document.save(doc_path)
            
        
# Plotting on Jupyter notebook    

    return fig
    
#--------------------------------------------------------------------------------------------------------------------

#------------------------------------------ PLOTTING BOX PLOT  -------------------------------------------------------

def plot_box(df, value='0', title=True, label = '',
             save_img=False, 
             save_path='', 
             save_doc=False,
             doc_name = 'first.docx' ):
    """
Description:

This function takes a DataFrame (df) as input and generates a box plot for a specified value column (value). The box plot can be customized with various options such as title, label, saving the plot image, saving the plot as a Word document, and specifying the save path and document name.

Parameters:

df (DataFrame): The input DataFrame containing the data to be plotted.
value (str, optional): The column name of the DataFrame to be plotted. Default is '0'.
title (bool, optional): Whether to display the plot title. Default is True.
label (str, optional): Additional label to be added to the plot title. Default is an empty string.
save_img (bool, optional): Whether to save the plot as an image. Default is False.
save_path (str, optional): The directory path where the plot image or document will be saved. Default is an empty string.
save_doc (bool, optional): Whether to save the plot as a Word document. Default is False.
doc_name (str, optional): The name of the Word document if save_doc is True. Default is 'first.docx'.

    """
    
    shape = None
    
    if value == '0':
        raise TypeError(" value can't be null")
    else:
       
        fig = go.Figure()
        keys = df.keys()
    
        for key in keys:
            fig.add_trace(go.Box(y = df[key][value], name=key))
        shape = str(df[key][value].shape[0])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        title_ = f' {label} Box Plot ({value}) ' + f'{shape} Segments' if title else ''
        fig.update_layout(
            yaxis_title= f"{value}",
            title = {
                'text': title_,
                'font': {
                    'family': 'Arial Bold',
                    'size': 15
                    }}, 
            title_x = 0.38,
            height = 500,
            width = 950,
            plot_bgcolor='white',
            xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            margin=dict(l=50, r=50, t=80, b=50
                       ))
        
# -------- SAVING IT AS A  Image------------

    if save_path and save_img:
        full_path = f"{save_path}/{title_}.png"
        pio.write_image(fig, full_path)  # Save the image   
        
        
# # -------- SAVING IT AS A DOCUMENT------------

#     if save_doc and save_path:
#         doc_path = os.path.join(save_path, doc_name)
#         temp_doc = None
#         if doc_path and os.path.exists(doc_path):
#             document = docx.Document(doc_path)
#             document.save(os.path.join(save_path, "temp.docx"))
#             temp_doc = docx.Document(os.path.join(save_path, "temp.docx"))
#             document = docx.Document()
#         else:
#             document = docx.Document()
            
            
#      # Create a new paragraph with the title
#         document.add_paragraph(f'{title_}') 
    
#         with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
#             image_path = tmpfile.name
#             pio.write_image(fig, image_path)
#             document.add_picture(image_path, width=Inches(4))
     
#         if temp_doc != None:
#             for element in temp_doc.element.body:
#                 document.element.body.append(element)
    
   
#             os.remove(os.path.join(save_path, "temp.docx"))

#         if doc_path is None:
#             document.save(f"{doc_name}.docx")
#         else:
#             document.save(doc_path)
            
            
                
    return fig
    
    
    
#--------------------------------------------------------------------------------------------------------------------    
    
    


def show_feature(features, value='max', title =''):
    n = np.array([])
    keys = []
    for key in features.keys():
        k = features[key][value].reshape(-1)
        keys.append(str(key))
        n = np.concatenate((n,k))
    
  
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=n, mode='lines'))
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, 170, 10),
            ticktext=keys,
            showline=True, 
            linewidth=2,
            linecolor='black', 
            mirror=True),
        yaxis=dict(title=f'{value}',
                    showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title_standoff=0,
                   mirror=True),
        width=1000,
        height= 400,
        title = {'text': f'{title}({value})',
                 'font': {'family': 'Arial Bold',
                          'size': 25 }},
        title_x = 0.5,
 
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    py.offline.iplot(fig)
import numpy as np
import plotly.graph_objects as go

from scipy.interpolate import interp1d

def envelope_plot(df, title="Envelope", 
                  xlabel='Time', 
                  ylabel='Amplitude', 
                  seg_num=1,
                  show_real=True,
                  save_img=False, 
                  save_path='', 
                  save_doc=False,
                  doc_name = 'first.docx',
                  sampling_freq = 12000):
    
    
    # Prepend the first value of (s) to the interpolating values
    s = df[seg_num - 1]
    u_x = [0]
    u_y = [s[0]]

    # Detect peaks and troughs and mark their location in u_x, u_y, l_x, l_y respectively
    for k in range(1, len(s) - 1):
        if (s[k] > s[k-1]) and (s[k] > s[k+1]):
            u_x.append(k)
            u_y.append(s[k])

    # Append the last value of (s) to the interpolating values
    u_x.append(len(s) - 1)
    u_y.append(s[-1])

    # Fit a model to the (u) value pairs
    u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0.0)

    # Evaluate the upper envelope model over the domain of (s)
    q_u = u_p(np.arange(len(s)))
#     mask = q_u <= 0.02
#     q_u[mask] = s.mean() + s[mask]*0.2
    
    fig = go.Figure()
    if show_real:
        fig.add_trace(go.Scatter(
            x= np.arange(0, 1, 1/sampling_freq),
            y=s,
            mode='lines',
            name= title + f'vib | (Seg{seg_num})',
        ))

    # Add trace for the upper envelope
    fig.add_trace(go.Scatter(
       x= np.arange(0, 1, 1/sampling_freq),
        y=q_u,
        mode='lines',
        name=title + f'env | (Seg{seg_num})',
        line=dict(color='red')
    ))
    
    fig.update_xaxes(title_text= xlabel+"(s)",
                         showgrid=True, 
                         gridcolor='lightgray',)
        
    fig.update_yaxes(title_text=ylabel, 
                         showgrid=True, 
                         gridcolor='lightgray',)
 
    fig.update_layout(title = {
                        'text': title + f' | (Segment {seg_num})',
                        'font': {
                        'family': 'Arial Bold',
                         'size': 25
                         }}, 
                     title_x = 0.4,
                     height= 500,
                     width=950,
                     plot_bgcolor='white',
                     showlegend=False,
                     xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
                     yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
                     margin=dict(l=50, r=20, t=100, b=50)
                     )
    title_ = title
    # -------- SAVING IT AS A  Image------------

    if save_path and save_img:
        full_path = f"{save_path}/{title_}.png"
        pio.write_image(fig, full_path)  # Save the image   
        
        
# -------- SAVING IT AS A DOCUMENT------------

    # if save_doc and save_path:
    #     doc_path = os.path.join(save_path, doc_name)
    #     temp_doc = None
    #     if doc_path and os.path.exists(doc_path):
    #         document = docx.Document(doc_path)
    #         document.save(os.path.join(save_path, "temp.docx"))
    #         temp_doc = docx.Document(os.path.join(save_path, "temp.docx"))
    #         document = docx.Document()
    #     else:
    #         document = docx.Document()
            
            
     # Create a new paragraph with the title
        # document.add_paragraph(f'{title_}') 
    
        # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        #     image_path = tmpfile.name
        #     pio.write_image(fig, image_path)
        #     document.add_picture(image_path, width=Inches(4))
     
        # if temp_doc != None:
        #     for element in temp_doc.element.body:
        #         document.element.body.append(element)

   
        #     os.remove(os.path.join(save_path, "temp.docx"))

        # if doc_path is None:
        #     document.save(f"{doc_name}.docx")
        # else:
        #     document.save(doc_path)
            
            
                
    return fig