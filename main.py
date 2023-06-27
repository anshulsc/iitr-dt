import streamlit as st
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sb
import plotly.offline as py
from Plotvib import make_segments, convert_df, plot_fft
from Plotvib import plot_data, filter_freq, plot_box
from Plotvib import extract_features, plot_features, envelope_plot
import scipy.fft as sp
import plotly.colors

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
    'rgb(214, 39, 40)'    # Red (same as above for emphasis)
]

# Initialize the app
# Configure page settings
extracted_files = []
extracted_file_names = []

st.set_page_config(layout="wide")
box = st.container()


# helper functions
def check_file(files, str):
            for file in files:
                if not file.endswith(str):
                    return False
            return True


def read_mat(file):
    data = scipy.io.loadmat(file)
    return data


def read_csv(file):
    data = pd.read_csv(file)
    return data


def process_file(file, col):
    file_extension = file.name.split(".")[-1]

    if file_extension == "mat":
        data = read_mat(file)
        keys = list(data.keys())
        if len(keys) > 0:

            selected_key = col.selectbox("Select a key", keys)
            selected_data = np.array(data[selected_key]).reshape(-1)
            return { f"{file.name.split('.')[0]}": selected_data}
        
        else:
            col.write("MAT File does not contain any keys.")

    elif file_extension == "csv":
        data = read_csv(file)
        if int(data.shape[1]) == 2:
            selected_data_x = np.array(data.iloc[:, 0]).reshape(-1)
            selected_data_y = np.array(data.iloc[:,1]).reshape(-1)
            return {f"{ file.name.split('.')[0]}_x": selected_data_x,
                    f"{file.name.split('.')[0]}_y" : selected_data_y,
                    }
        else:
             selected_data = np.array(data.iloc[:, 0]).reshape(-1)
             return {f"{ file.name.split('.')[0]}": selected_data}

    else:
        col.error(
            f"Invalid file format: {file_extension}. Only MAT files are supported."
        )





def plot_time(
    extracted_files, 
    index,
    div, 
    total_segment, 
    seg_num ,
    env_plot, isolated,
    sampling_frequency
):
    figs = []
    fig = None
    selected_name =   list(index)
    for k in selected_name:
            a = make_segments(extracted_files[k], total_segments=total_segment)
            if env_plot:
                fig = envelope_plot(df=a, title=k, seg_num=seg_num, show_real= (not isolated), sampling_freq = sampling_frequency)
                figs.append(fig) 
            else:
                fig = plot_data(df=a, title=k, seg_num=seg_num , sampling_freq = sampling_frequency)
                figs.append(fig)

    default_colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    combined_fig = go.Figure()
    if not env_plot:
        for i, fig in enumerate(figs):
            for  trace in fig.data:
                combined_fig.add_trace(trace.update(line=dict(color=default_colors[i]),showlegend=True, name =trace.name)  )
    else:
        for i, fig in enumerate(figs):
                if not isolated:
                    combined_fig.add_trace(fig.data[0].update(line=dict(color=default_colors[i]) , showlegend= False, name =fig.data[0].name))
                combined_fig.add_trace(fig.data[-1].update(line=dict(color= plotly_colors[-(i + 1)]), showlegend= True, name =fig.data[-1].name))

   
    combined_fig.update_layout(
        title="Time Domain",
        title_x=0.4,
        xaxis_title="Time",
        yaxis_title="Amplitude",
        legend_title="Legend",
        width= 1000,
        plot_bgcolor='white',
        showlegend=True,
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        margin=dict(l=50, r=20, t=80, b=0),
       )
    
    div.plotly_chart(combined_fig)
    div.markdown("---")

        


def plot_freq(
    extracted_files,
    index,
    filtered_freq,
    div,
    freq_range,
    total_segments,
    seg_num,
    limit,
    sampling_frequency,
):
    
    selected_name = [index]
    frequency = abs(sp.fftfreq(( extracted_files[selected_name[-1]].shape[0] // total_segments), 1 / sampling_frequency))
    steps = frequency[1] - frequency[0]
   
  
    for k in selected_name:
                
                a = make_segments(extracted_files[k], total_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, total_segments=total_segments)
                if filtered_freq:
                    c = filter_freq(b, amp=limit)
                    fig = plot_fft(
                            x_axis=frequency[: int(freq_range * (steps**-1))],
                            y_axis=c,
                            title=k,
                            seg_num=seg_num,
                            )
                    div.plotly_chart(fig)

                else:
                    fig = plot_fft(
                        x_axis=frequency[: int(freq_range * (steps**-1))],
                        y_axis=b,
                        title=k,
                        seg_num=seg_num,)
                    
                    div.plotly_chart(fig)

  

def plot_feat(
    extracted_files,
    index,
    filtered_freq,
    stats_features,
    div,
    total_segments,
    sampling_freq,
):
    keys = stats_features
    selected_name = list(index)
    # name = [extracted_file_names[0][k] for k in ix]
    extracted = {}
    try:
        for k in selected_name:
                a = make_segments(extracted_files[k], total_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, total_segments=total_segments)
                if filtered_freq:
                     c = filter_freq(b,file_name=k)
                     d = extract_features(c)
                     extracted[k] = d
                else:
                    d = extract_features(b)
                    extracted[k] = d   
    except:
        st.error("Please select a right key")
    
    if len(keys) > 0:
        div.plotly_chart(plot_features(extracted, keys=[f"{keys}"]))
    else:
        div.plotly_chart(plot_features(extracted))


def box_plot(
    extracted_files,
    index,
    filtered_freq,
    stats_features,
    div,
    total_segments,
    sampling_freq,
):
    keys = stats_features.lower()
    selected_name = list(index)
    extracted = {}
    try:
        for k in selected_name:
                a = make_segments(extracted_files[k], total_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, total_segments=total_segments)
                if filtered_freq:
                     c = filter_freq(b,file_name=k)
                     d = extract_features(c)
                     extracted[k] = d
                else:
                    d = extract_features(b)
                    extracted[k] = d   
    except:
        st.error("Please select a right key")

    if keys:
        div.plotly_chart(plot_box(extracted, value=keys))
    else:
        div.plotly_chart(plot_box(extracted, value="mean"))


def plot_scatter(
    extracted_files,
    index,
    div,
    total_segment,
    sampling_frequency,
    seg_num=1,
):
    combined_fig = go.Figure()
    selected_name = list(index)
    default_colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    for i, k in enumerate(selected_name):
            a = make_segments(extracted_files[k], total_segments=total_segment)
            combined_fig.add_trace(go.Scatter(
                                 y=a[seg_num - 1],
                                 mode = 'markers',
                                 marker=dict(color=default_colors[i]),
                                 name=f'{k} |  Segment {seg_num}',
                                 showlegend=True))
                                
        
   
    combined_fig.update_layout(
        title="Time Domain",
        title_x=0.4,
        xaxis_title="Data Points",
        yaxis_title="Amplitude",
        legend_title="Legend",
        width= 1000,
        plot_bgcolor='white',
        showlegend=True,
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        margin=dict(l=50, r=20, t=80, b=0),
       )
    
    div.plotly_chart(combined_fig)
    div.markdown("---")


#############################-----Enigne Starts Here-----#####################################


def main():

    df_seg ={}

    col_l, col_r = box.columns([0.4, 0.6], gap="large")
    col_l.title("File Upload")

    uploaded_files = col_l.file_uploader(
    
        "Upload MAT or CSV files", accept_multiple_files=True
    )
    
    exp = col_l.expander("Select Keys", expanded=True)
    cols = [] # columns for each file key
    li = [] # list of file names
    r = [] # list of keys

    if uploaded_files:
        all_files = [file.name for file in uploaded_files]

        # check if all files are .mat or .csv
        if check_file(all_files, ".mat"):
            if len(uploaded_files) <= 4 and len(uploaded_files) > 0:
                cols = exp.columns(len(uploaded_files))

            else:
                i = len(uploaded_files) // 4 + 1
                for _ in range(i):
                    cols.extend(exp.columns(4))

            for file, col1 in zip(uploaded_files, cols):
                dic = process_file(file, col1)
                df_seg.update(dic)

        elif check_file(all_files, ".csv"):
            for file in  uploaded_files :
                dic = process_file(file, col_l)
                df_seg.update(dic)
        
        else:
            st.error("Please upload only CSV or Mat files.")

    # Getting all file names
    extracted_file_names.append(li)
    extracted_file_names.append(r)
    
    
    # MainWWork here
    if uploaded_files:

         # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        # Total Number of Segments : Slider
        total_segments      =  col_l.number_input("Total Number of Segments", 1, int(10e6), 10)

        # Sampling Frequency : Slider
        sampling_frequency  =  col_l.number_input("Sampling Frequency",       1, int(10e6), 12000)

        col3, col4  =  col_l.columns(2)

        domain = col3.selectbox( "What to plot", ["", "Time Domain", "Frequency Domain", "Features", "Box Plot", "Scatter Plot"], )
         # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


        #------------   Time  ----------------------#
        if domain == "Time Domain":

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

            # File Name : MultiSelect Box
            selected_file_name = col4.multiselect( "Select a file", list(df_seg.keys()) )

            # Envelope Plot : Check Box
            env_plot = col4.checkbox("Plot Envelope")
           
            # Segment Number : Slider
            seg_num = col3.number_input("Segment Number", 1, total_segments)

            # Isolate Envelope : Check Box to isolate envelope
            isolated = col4.checkbox("Isolate Envelope") if env_plot else None

           # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


            if selected_file_name:

                # Call plot_time function
                plot_time(
                    df_seg,
                    selected_file_name,
                    col_r,
                    total_segments,
                    seg_num,
                    env_plot,
                    isolated,
                    sampling_frequency
                )
            else:

                st.write("No file selected")

        #------------  Frequency   -----------------#
        if domain == "Frequency Domain":

             # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
            # File Name : Select Box
            selected_file_name = col4.selectbox( "Select a file", list(df_seg.keys()))

            # Frequency Range : Slider  
            frequency = abs(sp.fftfreq(( df_seg[selected_file_name].shape[0]// total_segments), 1 / sampling_frequency))

            # Steps : Slider
            steps = frequency[1] - frequency[0]

            # Segment Number : Number Input
            seg_num = col4.number_input("Segment Number", 1, total_segments)

            # Filter Frequency : Check Box
            filtered_freq = col4.checkbox("Filter Frequency")

            # Frequency Limit : Number Input
            limit = col4.number_input("Frequency Limit", 0.0,1.0,0.2,step = 0.01) if filtered_freq else None

            # Frequency Range : 
            freq_range = col3.slider( "Frequency Range", float(0), float(sampling_frequency/2),float(sampling_frequency/2),  step= steps)
             # >>>>>>>>>>>>>>>>>>>>>>>>>>>> UI PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

            if selected_file_name:


                plot_freq(
                    df_seg,
                    selected_file_name,
                    filtered_freq,
                    col_r,
                    freq_range,
                    total_segments,
                    seg_num,
                    limit,
                    sampling_frequency,
                )

            else:
                col_l.write("No file selected")

        #------------  Features  -----------------#
        if domain == "Features":

            selected_file_name = col4.multiselect( "Select file(s)", list(df_seg.keys()))
            # multiselct stats features but only 4 can be selected at a time

            stats_features = col4.selectbox( "Select stats features",
                [
                    "Mean",
                    "Max",
                    "Variance",
                    "Skewness",
                    "Kurtosis",
                    "shape_factor",
                    "impulse_factor",
                ],
                key="stats_features",
            )

            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time

            if selected_file_name:
                
                plot_feat(
                    df_seg,
                    selected_file_name,
                    filtered_freq,
                    stats_features,
                    col_r,
                    total_segments,
                    sampling_frequency,
                )
            else:
                st.write("No file selected")

        #------------  Box Plot  -----------------#
        if domain == "Box Plot":

            selected_file_name = col4.multiselect( "Select file(s)", list(df_seg.keys()))

            # multiselct stats features but only 4 can be selected at a time
            stats_features = col4.selectbox( "Select stats features",  [ "Mean", "Max", "Variance", "Skewness", "Kurtosis", "shape_factor","impulse_factor"],)
            
            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time

            if selected_file_name:
                

                box_plot(
                    df_seg,
                    selected_file_name,
                    filtered_freq,
                    stats_features,
                    col_r,
                    total_segments,
                    sampling_frequency,
                )
            else:
                st.write("No file selected")

        #------------ Scatter Plot -----------------#
        if domain == "Scatter Plot":

            # multiselct stats features but only 4 can be selected at a time
            selected_file_name = col4.multiselect( "Select file(s)", list(df_seg.keys()))

            if selected_file_name:

               
                
                plot_scatter(
                    extracted_files=df_seg,
                    index=selected_file_name,
                    div = col_r,
                    total_segment=total_segments,
                    sampling_frequency=sampling_frequency,
                    seg_num=1,
                )
            else:
                st.write("No file selected")

        with col_l:
            if col_l.button("Show Files"):
                st.write("Extracted Files:", df_seg)
                st.write("Extracted File Names:", {key : val.shape for key, val in df_seg.items()})


        # Perform actions with extracted_files and extracted_file_names
        # Example: Save files or perform further processing

        # col_l.markdown(f"Extracted Files: {extracted_files}")
        # col_l.markdown("Extracted File Names:", extracted_file_names)


if __name__ == "__main__":
    main()
