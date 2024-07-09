import numpy as np
from PIL import Image


class UI:
    about_block = """

    ### About

    This is a charactor-creating app. Give image of your pet.

    """

    css = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1400px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: "#111";
        background-color: "#eee";
    }}
</style>
"""


headers = {"Content-Type": "application/json"}
