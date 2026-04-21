import glob
import json
import os
import uuid

from mn_fibril_modeller_gemmi.viewer.molstar_custom_component.dataclasses import ChainVisualization, StructureVisualization


parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend", "build")

_component_func = None
_static_files = None


def molstar_custom_component(
    structures: list[StructureVisualization],
    key: str | None,
    height: str | int = "500px",
    width: str | int = "100%",
    show_controls: bool = False,
    selection_mode: bool = False,
    download_filename: str | None = None,
    html_filename: str | None = None,
    force_reload: bool = False,
):
    global _component_func
    if _component_func is None:
        import streamlit.components.v1 as components

        _component_func = components.declare_component("molstar_custom_component", path=build_dir)

    serialized_structures = json.dumps([structure.to_dict() for structure in structures])

    component_value = _component_func(
        structures=serialized_structures,
        height=f"{height}px" if isinstance(height, int) else height,
        width=f"{width}px" if isinstance(width, int) else width,
        showControls=show_controls,
        selectionMode=selection_mode,
        forceReload=force_reload,
        key=key,
    )

    if download_filename:
        import streamlit as st

        for structure in structures:
            st.download_button("Download PDB", data=structure.pdb, file_name=f"{download_filename}.pdb")
            break

    if html_filename:
        import streamlit as st

        st.download_button(
            "Download HTML",
            data=molstar_html(structures),
            file_name=f"{html_filename}.html",
            mime="text/html",
        )

    return component_value


def read_static_files():
    global _static_files
    if _static_files is None:
        css_files = sorted(glob.glob(os.path.join(build_dir, "chunk-*.css")))
        if not css_files:
            raise FileNotFoundError("No CSS file found in the Mol* build directory.")
        js_files = sorted(glob.glob(os.path.join(build_dir, "chunk-*.js")))
        if not js_files:
            raise FileNotFoundError("No JS file found in the Mol* build directory.")
        with open(css_files[0], encoding="utf-8") as handle:
            css_content = handle.read()
        with open(js_files[0], encoding="utf-8") as handle:
            js_content = handle.read()
        _static_files = css_content, js_content
    return _static_files


def molstar_html(structures: list[StructureVisualization]):
    css_content, js_content = read_static_files()
    return f"""
            <div id="root" class="molstar_notebook"></div>
            <style type="text/css">
            {css_content}
            </style>
            <script>
            window.STRUCTURES = {json.dumps([s.to_dict() for s in structures])};

            {js_content.replace('</script>"', '</" + "script>"')}
            </script>
            """


def molstar_notebook(structures: list[StructureVisualization], height="500px", width="800px"):
    from IPython.display import HTML, Javascript, display

    html_data = molstar_html(structures).strip()
    assert html_data.endswith("</script>"), "Molstar HTML should end with </script>"

    wrapper_id = f"molstar_{uuid.uuid4()}"
    js_code = f"""
    setTimeout(function(){{
        var wrapper = document.getElementById("{wrapper_id}");
        if (wrapper === null) {{
            throw new Error("Wrapper element #{wrapper_id} not found anymore");
        }}
        var blob = new Blob([{json.dumps(html_data[:-1])} + ">"], {{ type: 'text/html' }});
        var url = URL.createObjectURL(blob);

        var iframe = document.createElement('iframe');
        iframe.src = url;
        iframe.style = "border: 0; width: {width}; height: {height}";
        iframe.allowFullscreen = true;
        wrapper.appendChild(iframe);
    }}, 100);
    """

    display(HTML(f'<div id="{wrapper_id}"></div>'))
    display(Javascript(js_code))


__all__ = ["molstar_custom_component", "molstar_html", "molstar_notebook", "StructureVisualization", "ChainVisualization"]
