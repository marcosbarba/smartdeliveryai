"""Punto de entrada `uv run sdai-app`: arranca la interfaz Streamlit.

`app.py` no sirve como entry point directo porque ejecuta codigo de Streamlit
(`st.set_page_config`, todo el renderizado de la pagina) en cuanto se importa — solo funciona
dentro del bootstrap propio de `streamlit run`, no como una funcion Python normal llamada desde
otro proceso. Este modulo no importa `app.py`: se limita a invocar `streamlit run app.py`,
simulando la CLI de Streamlit con `sys.argv`.

Uso:

    uv run sdai-app
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
