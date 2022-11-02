import asyncio
import base64
import hashlib
from io import BytesIO

import ipywidgets
from IPython.display import display, HTML

class MediaPlayer:
    def __init__(self, num_steps, frame_callback, frame_image_callback):
        self.frame_image_callback = frame_image_callback
        self.slider = ipywidgets.IntSlider(value=0, min=0, max=num_steps - 1, step=1)
        self.slider.observe(frame_callback, names="value")

        self.play_button = ipywidgets.Button(description="▶")
        self.play_button.on_click(self.on_play)

        self.save_button = ipywidgets.Button(description="save")
        self.save_button.on_click(self.on_save)
        self.save_button.disable = False

        self.save_file = ipywidgets.Text(value="sample.gif", description="save file name: ")

    def on_play(self, btn):
        if btn.description == "▶":
            btn.description = "❙❙"
            asyncio.ensure_future(self.play_loop(btn))
        else:
            btn.description = "▶"

    async def play_loop(self, btn):
        while True:
            self.slider.value += 1
            await asyncio.sleep(0.1)
            if btn.description == "▶":
                return 
            if self.slider.max == self.slider.value:
                btn.description = "▶"
                self.slider.value = 0
                return 

    def on_save(self, btn):
        if not btn.disable:
            btn.disable = True
            asyncio.ensure_future(self.save_loop(btn))

    async def save_loop(self, btn):
        imgs = []
        for i in range(self.slider.min, self.slider.max + 1):
            self.slider.value = i
            imgs.append(self.frame_image_callback())
        data = BytesIO()
        imgs[0].save(data, "gif", save_all=True, append_images=imgs[1:], duration=100)
        b64 = base64.b64encode(data.getvalue())
        payload = b64.decode()
        digest = hashlib.md5(data.getvalue()).hexdigest()
        id = f'dl_{digest}'
        display(HTML(f"""
<html>
<body>
<a id="{id}" download="{self.save_file.value}" href="data:image/*;base64,{payload}" download>
</a>

<script>
(function download() {{
document.getElementById('{id}').click();
}})()
</script>

</body>
</html>
"""))
        btn.disable = False

    def show(self):
        display(self.slider, self.play_button, self.save_button, self.save_file)