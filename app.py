from layout import Layout
from functions_check import Functions
from buttons import Buttons

layout = Layout()
layout.set_background()
layout.sidebar_options()
layout.main_area()

functions = Functions(layout=layout)
buttons = Buttons(functions=functions)
buttons.render()