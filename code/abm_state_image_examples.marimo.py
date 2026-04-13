import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    from bokeh.plotting import figure, show, output_file, save
    from bokeh.models import ColumnDataSource, HoverTool
    import base64

    from spatialtissuepy import SpatialTissueData
    from spatialtissuepy.synthetic.physicell import (
        PhysiCellSimulation,
        PhysiCellTimeStep,
        read_physicell_timestep,
        discover_physicell_timesteps,
        is_alive,
        is_dead,
    )
    from spatialtissuepy.summary import StatisticsPanel, SpatialSummary
    from spatialtissuepy.viz import plot_spatial_scatter
    return (
        ColumnDataSource,
        HoverTool,
        PhysiCellSimulation,
        base64,
        figure,
        os,
        output_file,
        pd,
        save,
        show,
    )


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    base64,
    figure,
    os,
    output_file,
    pd,
    save,
    show,
):
    # 1. Load your data (mimicking the d3.csv loading)
    # Ensure your CSV has columns: x, y, cell_type
    # df = pd.read_csv("data/state_1_sample_data.csv") 
    # For demonstration, creating dummy data:
    data = {
        'x': [100, 250, 400, 550, 700, 850],
        'y': [100, 250, 400, 550, 700, 850],
        'cell_type': [
            "M0_macrophage", "M1_macrophage", "M2_macrophage", 
            "effector_T_cell", "exhausted_T_cell", "malignant_epithelial_cell"
        ]
    }
    df = pd.DataFrame(data)

    # 2. Map cell types to icon paths
    # NOTE: For local viewing, file paths usually need to be relative to the HTML 
    # or hosted on a local server due to browser security policies.
    icon_map = {
      "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
      "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
      "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
      "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
      "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
      "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
    }

    # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
    def image_to_base64(path):
        """
        Reads an image from disk and converts it to a base64 encoded data URI.
        Returns a transparent placeholder circle if the file is not found.
        """
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            # Return a simple SVG circle as a fallback so the plot doesn't break
            return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
    
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Determine mime type based on extension
            if path.lower().endswith('.svg'):
                mime = "image/svg+xml"
            else:
                mime = "image/png"
            return f"data:{mime};base64,{encoded_string}"

    # --- 4. PREPARE DATA ---
    # Map the 'cell_type' column to the real file paths
    df['file_path'] = df['cell_type'].map(icon_map)

    # Convert those file paths to Base64 strings
    print("Encoding images...")
    df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))

    source = ColumnDataSource(df)

    # --- 5. PLOTTING ---
    plot_width = 1000
    plot_height = 1000

    p = figure(width=plot_width, height=plot_height, 
               title="Cell Plot (Embedded Images)",
               match_aspect=True,
               tools="pan,wheel_zoom,reset,save")

    # Hide grid for a cleaner 'spatial biology' look
    p.xgrid.visible = False
    p.ygrid.visible = False

    # Plot the images
    # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
    p.image_url(url='icon_image', x='x', y='y', 
                w=40, h=40, w_units='screen', h_units='screen', 
                anchor='center', source=source)

    # Add Tooltips
    hover = HoverTool(tooltips=[
        ("Cell Type", "@cell_type"),
        ("Position", "(@x, @y)")
    ])
    p.add_tools(hover)

    # --- 6. SAVE AND SHOW ---
    output_filename = "output/figures/abm/state-snapshots/cell_plot_embedded.html"
    output_file(output_filename)

    print(f"Attempting to save to: {os.path.abspath(output_filename)}")
    save(p) # Explicitly saves the file
    print("Save successful.")

    # Try to open in browser automatically
    try:
        show(p)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open {output_filename} manually.")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    PhysiCellSimulation,
    base64,
    figure,
    os,
    output_file,
    save,
    show,
):
    def _():    
        # 1. Load your data (mimicking the d3.csv loading)
        sim014 = PhysiCellSimulation.from_output_folder(
            'data/abm/raw/sim_014/'
        )
        df = sim014.get_timestep(690).to_dataframe()    
    
        # 2. Map cell types to icon paths
        # NOTE: For local viewing, file paths usually need to be relative to the HTML 
        # or hosted on a local server due to browser security policies.
        icon_map = {
          "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
          "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
          "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
          "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
          "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
          "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
        }
    
        # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
        def image_to_base64(path):
            """
            Reads an image from disk and converts it to a base64 encoded data URI.
            Returns a transparent placeholder circle if the file is not found.
            """
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                # Return a simple SVG circle as a fallback so the plot doesn't break
                return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
        
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine mime type based on extension
                if path.lower().endswith('.svg'):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                return f"data:{mime};base64,{encoded_string}"
    
        # --- 4. PREPARE DATA ---
        # Map the 'cell_type' column to the real file paths
        df['file_path'] = df['cell_type'].map(icon_map)
    
        # Convert those file paths to Base64 strings
        print("Encoding images...")
        df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))
    
        source = ColumnDataSource(df)
    
        # --- 5. PLOTTING ---
        plot_width = 1000
        plot_height = 1000
    
        p = figure(width=plot_width, height=plot_height, 
                   title="Example Snapshot of State 1",
                   match_aspect=True,
                   tools="pan,wheel_zoom,reset,save")

        # Set the inner drawing area to white
        p.background_fill_color = "white"
        p.background_fill_alpha = 1.0  # Ensure it's opaque
    
        # Set the outer border area (around the axes/title) to white
        p.border_fill_color = "white"
        p.border_fill_alpha = 1.0
    
        # If you want to remove the outline border line as well:
        p.outline_line_color = None
    
        # Hide grid for a cleaner 'spatial biology' look
        p.xgrid.visible = False
        p.ygrid.visible = False
    
        # Plot the images
        # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
        p.image_url(url='icon_image', x='x', y='y', 
                    w=40, h=40, w_units='screen', h_units='screen', 
                    anchor='center', source=source)
    
        # Add Tooltips
        hover = HoverTool(tooltips=[
            ("Cell Type", "@cell_type"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
    
        # --- 6. SAVE AND SHOW ---
        output_filename = "output/figures/abm/state-snapshots/sim014_state1.html"
        output_file(output_filename)
    
        print(f"Attempting to save to: {os.path.abspath(output_filename)}")
        save(p) # Explicitly saves the file
        print("Save successful.")
    
        # Try to open in browser automatically
        try:
            show(p)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {output_filename} manually.")
    _()
    return


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    PhysiCellSimulation,
    base64,
    figure,
    os,
    output_file,
    save,
    show,
):
    def _():    
        # 1. Load your data (mimicking the d3.csv loading)
        sim014 = PhysiCellSimulation.from_output_folder(
            'data/abm/raw/sim_014/'
        )
        df = sim014.get_timestep(40).to_dataframe()    
    
        # 2. Map cell types to icon paths
        # NOTE: For local viewing, file paths usually need to be relative to the HTML 
        # or hosted on a local server due to browser security policies.
        icon_map = {
          "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
          "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
          "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
          "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
          "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
          "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
        }
    
        # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
        def image_to_base64(path):
            """
            Reads an image from disk and converts it to a base64 encoded data URI.
            Returns a transparent placeholder circle if the file is not found.
            """
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                # Return a simple SVG circle as a fallback so the plot doesn't break
                return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
        
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine mime type based on extension
                if path.lower().endswith('.svg'):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                return f"data:{mime};base64,{encoded_string}"
    
        # --- 4. PREPARE DATA ---
        # Map the 'cell_type' column to the real file paths
        df['file_path'] = df['cell_type'].map(icon_map)
    
        # Convert those file paths to Base64 strings
        print("Encoding images...")
        df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))
    
        source = ColumnDataSource(df)
    
        # --- 5. PLOTTING ---
        plot_width = 1000
        plot_height = 1000
    
        p = figure(width=plot_width, height=plot_height, 
                   title="Example Snapshot of State 2",
                   match_aspect=True,
                   tools="pan,wheel_zoom,reset,save")

        # Set the inner drawing area to white
        p.background_fill_color = "white"
        p.background_fill_alpha = 1.0  # Ensure it's opaque
    
        # Set the outer border area (around the axes/title) to white
        p.border_fill_color = "white"
        p.border_fill_alpha = 1.0
    
        # If you want to remove the outline border line as well:
        p.outline_line_color = None
    
        # Hide grid for a cleaner 'spatial biology' look
        p.xgrid.visible = False
        p.ygrid.visible = False
    
        # Plot the images
        # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
        p.image_url(url='icon_image', x='x', y='y', 
                    w=40, h=40, w_units='screen', h_units='screen', 
                    anchor='center', source=source)
    
        # Add Tooltips
        hover = HoverTool(tooltips=[
            ("Cell Type", "@cell_type"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
    
        # --- 6. SAVE AND SHOW ---
        output_filename = "output/figures/abm/state-snapshots/sim014_state2.html"
        output_file(output_filename)
    
        print(f"Attempting to save to: {os.path.abspath(output_filename)}")
        save(p) # Explicitly saves the file
        print("Save successful.")
    
        # Try to open in browser automatically
        try:
            show(p)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {output_filename} manually.")
    _()
    return


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    PhysiCellSimulation,
    base64,
    figure,
    os,
    output_file,
    save,
    show,
):
    def _():    
        # 1. Load your data (mimicking the d3.csv loading)
        sim014 = PhysiCellSimulation.from_output_folder(
            'data/abm/raw/sim_000/'
        )
        df = sim014.get_timestep(50).to_dataframe()    
    
        # 2. Map cell types to icon paths
        # NOTE: For local viewing, file paths usually need to be relative to the HTML 
        # or hosted on a local server due to browser security policies.
        icon_map = {
          "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
          "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
          "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
          "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
          "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
          "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
        }
    
        # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
        def image_to_base64(path):
            """
            Reads an image from disk and converts it to a base64 encoded data URI.
            Returns a transparent placeholder circle if the file is not found.
            """
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                # Return a simple SVG circle as a fallback so the plot doesn't break
                return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
        
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine mime type based on extension
                if path.lower().endswith('.svg'):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                return f"data:{mime};base64,{encoded_string}"
    
        # --- 4. PREPARE DATA ---
        # Map the 'cell_type' column to the real file paths
        df['file_path'] = df['cell_type'].map(icon_map)
    
        # Convert those file paths to Base64 strings
        print("Encoding images...")
        df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))
    
        source = ColumnDataSource(df)
    
        # --- 5. PLOTTING ---
        plot_width = 1000
        plot_height = 1000
    
        p = figure(width=plot_width, height=plot_height, 
                   title="Example Snapshot of State 3",
                   match_aspect=True,
                   tools="pan,wheel_zoom,reset,save")

        # Set the inner drawing area to white
        p.background_fill_color = "white"
        p.background_fill_alpha = 1.0  # Ensure it's opaque
    
        # Set the outer border area (around the axes/title) to white
        p.border_fill_color = "white"
        p.border_fill_alpha = 1.0
    
        # If you want to remove the outline border line as well:
        p.outline_line_color = None
    
        # Hide grid for a cleaner 'spatial biology' look
        p.xgrid.visible = False
        p.ygrid.visible = False
    
        # Plot the images
        # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
        p.image_url(url='icon_image', x='x', y='y', 
                    w=40, h=40, w_units='screen', h_units='screen', 
                    anchor='center', source=source)
    
        # Add Tooltips
        hover = HoverTool(tooltips=[
            ("Cell Type", "@cell_type"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
    
        # --- 6. SAVE AND SHOW ---
        output_filename = "output/figures/abm/state-snapshots/sim000_state3.html"
        output_file(output_filename)
    
        print(f"Attempting to save to: {os.path.abspath(output_filename)}")
        save(p) # Explicitly saves the file
        print("Save successful.")
    
        # Try to open in browser automatically
        try:
            show(p)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {output_filename} manually.")
    _()
    return


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    PhysiCellSimulation,
    base64,
    figure,
    os,
    output_file,
    save,
    show,
):
    def _():    
        # 1. Load your data (mimicking the d3.csv loading)
        sim = PhysiCellSimulation.from_output_folder(
            'data/abm/raw/sim_000/'
        )
        df = sim.get_timestep(690).to_dataframe()    
    
        # 2. Map cell types to icon paths
        # NOTE: For local viewing, file paths usually need to be relative to the HTML 
        # or hosted on a local server due to browser security policies.
        icon_map = {
          "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
          "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
          "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
          "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
          "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
          "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
        }
    
        # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
        def image_to_base64(path):
            """
            Reads an image from disk and converts it to a base64 encoded data URI.
            Returns a transparent placeholder circle if the file is not found.
            """
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                # Return a simple SVG circle as a fallback so the plot doesn't break
                return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
        
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine mime type based on extension
                if path.lower().endswith('.svg'):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                return f"data:{mime};base64,{encoded_string}"
    
        # --- 4. PREPARE DATA ---
        # Map the 'cell_type' column to the real file paths
        df['file_path'] = df['cell_type'].map(icon_map)
    
        # Convert those file paths to Base64 strings
        print("Encoding images...")
        df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))
    
        source = ColumnDataSource(df)
    
        # --- 5. PLOTTING ---
        plot_width = 1000
        plot_height = 1000
    
        p = figure(width=plot_width, height=plot_height, 
                   title="Example Snapshot of State 6",
                   match_aspect=True,
                   tools="pan,wheel_zoom,reset,save")

        # Set the inner drawing area to white
        p.background_fill_color = "white"
        p.background_fill_alpha = 1.0  # Ensure it's opaque
    
        # Set the outer border area (around the axes/title) to white
        p.border_fill_color = "white"
        p.border_fill_alpha = 1.0
    
        # If you want to remove the outline border line as well:
        p.outline_line_color = None
    
        # Hide grid for a cleaner 'spatial biology' look
        p.xgrid.visible = False
        p.ygrid.visible = False
    
        # Plot the images
        # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
        p.image_url(url='icon_image', x='x', y='y', 
                    w=40, h=40, w_units='screen', h_units='screen', 
                    anchor='center', source=source)
    
        # Add Tooltips
        hover = HoverTool(tooltips=[
            ("Cell Type", "@cell_type"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
    
        # --- 6. SAVE AND SHOW ---
        output_filename = "output/figures/abm/state-snapshots/sim000_state6.html"
        output_file(output_filename)
    
        print(f"Attempting to save to: {os.path.abspath(output_filename)}")
        save(p) # Explicitly saves the file
        print("Save successful.")
    
        # Try to open in browser automatically
        try:
            show(p)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {output_filename} manually.")
    _()
    return


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    PhysiCellSimulation,
    base64,
    figure,
    os,
    output_file,
    save,
    show,
):
    def _():    
        # 1. Load your data (mimicking the d3.csv loading)
        sim = PhysiCellSimulation.from_output_folder(
            'data/abm/raw/sim_003/'
        )
        df = sim.get_timestep(450).to_dataframe()    
    
        # 2. Map cell types to icon paths
        # NOTE: For local viewing, file paths usually need to be relative to the HTML 
        # or hosted on a local server due to browser security policies.
        icon_map = {
          "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
          "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
          "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
          "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
          "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
          "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
        }
    
        # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
        def image_to_base64(path):
            """
            Reads an image from disk and converts it to a base64 encoded data URI.
            Returns a transparent placeholder circle if the file is not found.
            """
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                # Return a simple SVG circle as a fallback so the plot doesn't break
                return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
        
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine mime type based on extension
                if path.lower().endswith('.svg'):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                return f"data:{mime};base64,{encoded_string}"
    
        # --- 4. PREPARE DATA ---
        # Map the 'cell_type' column to the real file paths
        df['file_path'] = df['cell_type'].map(icon_map)
    
        # Convert those file paths to Base64 strings
        print("Encoding images...")
        df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))
    
        source = ColumnDataSource(df)
    
        # --- 5. PLOTTING ---
        plot_width = 1000
        plot_height = 1000
    
        p = figure(width=plot_width, height=plot_height, 
                   title="Example Snapshot of State 4",
                   match_aspect=True,
                   tools="pan,wheel_zoom,reset,save")

        # Set the inner drawing area to white
        p.background_fill_color = "white"
        p.background_fill_alpha = 1.0  # Ensure it's opaque
    
        # Set the outer border area (around the axes/title) to white
        p.border_fill_color = "white"
        p.border_fill_alpha = 1.0
    
        # If you want to remove the outline border line as well:
        p.outline_line_color = None
    
        # Hide grid for a cleaner 'spatial biology' look
        p.xgrid.visible = False
        p.ygrid.visible = False
    
        # Plot the images
        # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
        p.image_url(url='icon_image', x='x', y='y', 
                    w=40, h=40, w_units='screen', h_units='screen', 
                    anchor='center', source=source)
    
        # Add Tooltips
        hover = HoverTool(tooltips=[
            ("Cell Type", "@cell_type"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
    
        # --- 6. SAVE AND SHOW ---
        output_filename = "output/figures/abm/state-snapshots/sim003_state4.html"
        output_file(output_filename)
    
        print(f"Attempting to save to: {os.path.abspath(output_filename)}")
        save(p) # Explicitly saves the file
        print("Save successful.")
    
        # Try to open in browser automatically
        try:
            show(p)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {output_filename} manually.")
    _()
    return


@app.cell
def _(
    ColumnDataSource,
    HoverTool,
    PhysiCellSimulation,
    base64,
    figure,
    os,
    output_file,
    save,
    show,
):
    def _():    
        # 1. Load your data (mimicking the d3.csv loading)
        sim = PhysiCellSimulation.from_output_folder(
            'data/abm/raw/sim_003/'
        )
        df = sim.get_timestep(250).to_dataframe()    
    
        # 2. Map cell types to icon paths
        # NOTE: For local viewing, file paths usually need to be relative to the HTML 
        # or hosted on a local server due to browser security policies.
        icon_map = {
          "M0_macrophage": "docs/biorender_icons/icons/png/m0.png",
          "M1_macrophage": "docs/biorender_icons/icons/png/m1.png",
          "M2_macrophage": "docs/biorender_icons/icons/png/m2.png",
          "effector_T_cell": "docs/biorender_icons/icons/png/cd8_effector.png",
          "exhausted_T_cell": "docs/biorender_icons/icons/png/cd8_exhausted.png",
          "malignant_epithelial_cell": "docs/biorender_icons/icons/png/malignant_epithelial_cell.png"
        }
    
        # --- 3. HELPER FUNCTION: IMAGE TO BASE64 ---
        def image_to_base64(path):
            """
            Reads an image from disk and converts it to a base64 encoded data URI.
            Returns a transparent placeholder circle if the file is not found.
            """
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                # Return a simple SVG circle as a fallback so the plot doesn't break
                return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
        
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine mime type based on extension
                if path.lower().endswith('.svg'):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                return f"data:{mime};base64,{encoded_string}"
    
        # --- 4. PREPARE DATA ---
        # Map the 'cell_type' column to the real file paths
        df['file_path'] = df['cell_type'].map(icon_map)
    
        # Convert those file paths to Base64 strings
        print("Encoding images...")
        df['icon_image'] = df['file_path'].apply(lambda x: image_to_base64(str(x)))
    
        source = ColumnDataSource(df)
    
        # --- 5. PLOTTING ---
        plot_width = 1000
        plot_height = 1000
    
        p = figure(width=plot_width, height=plot_height, 
                   title="Example Snapshot of State 4",
                   match_aspect=True,
                   tools="pan,wheel_zoom,reset,save")

        # Set the inner drawing area to white
        p.background_fill_color = "white"
        p.background_fill_alpha = 1.0  # Ensure it's opaque
    
        # Set the outer border area (around the axes/title) to white
        p.border_fill_color = "white"
        p.border_fill_alpha = 1.0
    
        # If you want to remove the outline border line as well:
        p.outline_line_color = None
    
        # Hide grid for a cleaner 'spatial biology' look
        p.xgrid.visible = False
        p.ygrid.visible = False
    
        # Plot the images
        # w/h_units='screen' keeps icons constant size in pixels regardless of zoom
        p.image_url(url='icon_image', x='x', y='y', 
                    w=40, h=40, w_units='screen', h_units='screen', 
                    anchor='center', source=source)
    
        # Add Tooltips
        hover = HoverTool(tooltips=[
            ("Cell Type", "@cell_type"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
    
        # --- 6. SAVE AND SHOW ---
        output_filename = "output/figures/abm/state-snapshots/sim003_state4.html"
        output_file(output_filename)
    
        print(f"Attempting to save to: {os.path.abspath(output_filename)}")
        save(p) # Explicitly saves the file
        print("Save successful.")
    
        # Try to open in browser automatically
        try:
            show(p)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open {output_filename} manually.")
    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
