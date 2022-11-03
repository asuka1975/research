import sys
import math

import networkx as nx
from bokeh.plotting import figure, from_networkx
from bokeh.io import push_notebook
from bokeh.io.export import get_screenshot_as_png
from bokeh.models import Range1d, Circle, ColumnDataSource, Arrow, NormalHead, Scatter
from bokeh.palettes import RdBu11

import glfw

from ..archive.observe import Observe

from .widgets import MediaPlayer

from OpenGL.GL import *
import glfw
import numpy as np
from ctypes import Structure, sizeof
import sys

class Shader:
    def __init__(self):
        self.handle = glCreateProgram()
    
    def attach_shader(self, content, type, log_always=False):
        shader = glCreateShader(type)
        glShaderSource(shader, [content])
        glCompileShader(shader)

        status = ctypes.c_uint(GL_UNSIGNED_INT)
        glGetShaderiv(shader, GL_COMPILE_STATUS, status)
        if log_always or not status:
            try:
                print(glGetShaderInfoLog(shader).decode("utf-8"), file=sys.stderr)
            except:
                print(glGetShaderInfoLog(shader), file=sys.stderr)
            glDeleteShader(shader)
            return False
        
        glAttachShader(self.handle, shader)
        glDeleteShader(shader)
        return True

    def link(self, log_always=False):
        glLinkProgram(self.handle)
        status = ctypes.c_uint(GL_UNSIGNED_INT)
        glGetProgramiv(self.handle, GL_LINK_STATUS, status)
        if log_always or not status:
            try:
                print(glGetProgramInfoLog(self.handle).decode("utf-8"), file=sys.stderr)
            except:
                print(glGetProgramInfoLog(self.handle), file=sys.stderr)
            return False
        return True
    
    def use(self):
        glUseProgram(self.handle)

    def unuse(self):
        glUseProgram(0)

class Node(Structure):
    _fields_ = [
        ("x", GLfloat),
        ("y", GLfloat)
    ]

class Edge(Structure): 
    _fields_ = [
        ("in_", GLuint),
        ("out", GLuint),
        ("weight", GLfloat)
    ]

node_vert = """#version 460

layout (location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0, 1);
}
"""

node_frag = """#version 460

layout (location = 0) out vec4 outColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(length(coord) > 0.5) {
        discard;
    }
    
    outColor = vec4(vec3(0.3), 1);
}
"""

edge_vert = """#version 460

layout (location = 0) in uint inIn;
layout (location = 1) in uint inOut;
layout (location = 2) in float weight;

layout (location = 0) out uint outIn;
layout (location = 1) out uint outOut;
layout (location = 2) out float outWeight;

void main() {
    outIn = inIn;
    outOut = inOut;
    outWeight = weight;
    gl_Position = vec4(weight, weight, 0, 1);
}
"""

edge_geom = """#version 460

layout (points) in;
layout (triangle_strip, max_vertices = 10) out;

layout (location = 0) in uint inIn[];
layout (location = 1) in uint inOut[];
layout (location = 2) in float weight[];

layout (location = 0) out float outWeight;

struct Node {
    float x;
    float y;
};

layout (std430, binding = 0) buffer nodes {
    Node nd[];
};

const float PI = 3.14159265359;

vec2 rotate2d(vec2 v, float t) {
    return mat2(
        cos(t), -sin(t),
        sin(t), cos(t)
    ) * v;
}

float gauss(float v) {
    return -exp(-v * v);
}

void main() {
    float width = 0.005 * (1 - gauss(weight[0]));
    vec2 in_v = vec2(nd[inIn[0]].x, nd[inIn[0]].y);
    vec2 out_v = vec2(nd[inOut[0]].x, nd[inOut[0]].y);

    gl_Position = vec4(in_v + width * rotate2d(out_v - in_v, PI / 2), 0, 1);
    outWeight = weight[0];
    EmitVertex();
    gl_Position = vec4(in_v + width * rotate2d(out_v - in_v, -PI / 2), 0, 1);
    outWeight = weight[0];
    EmitVertex();
    gl_Position = vec4(out_v - 0.05 * cos(PI / 4) * normalize(out_v - in_v) + width * rotate2d(out_v - in_v, PI / 2), 0, 1);
    outWeight = weight[0];
    EmitVertex();
    gl_Position = vec4(out_v - 0.05 * cos(PI / 4) * normalize(out_v - in_v) + width * rotate2d(out_v - in_v, -PI / 2), 0, 1);
    outWeight = weight[0];
    EmitVertex();

    // Arrow Head
    gl_Position = vec4(rotate2d(-0.05 * normalize(out_v - in_v), -PI / 6) + out_v, 0, 1);
    outWeight = weight[0];
    EmitVertex();
    gl_Position = vec4(out_v, 0, 1);
    outWeight = weight[0];
    EmitVertex();
    gl_Position = vec4(rotate2d(-0.05 * normalize(out_v - in_v), PI / 6) + out_v, 0, 1);
    outWeight = weight[0];
    EmitVertex();
    
    EndPrimitive();
}
"""

edge_frag = """#version 460

layout (location = 0) in float weight;

layout (location = 0) out vec4 outColor;

void main() {
    if(weight < 0) {
        outColor = vec4(0, 0, 1, 0.5);
    } else {
        outColor = vec4(1, 0, 0, 0.5);
    }
}
"""

size = (1000, 1000)

class GLNetwork:
    def __init__(self, network_data):
        glfw.init()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.SAMPLES, 16)
        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(*size, "sample", None, None)
        glfw.make_context_current(self.window)

        self.data = network_data

        self.dw = 0.1
        self.xmin = min([n["x"] for n in self.data["task0"][0]["node"]]) - self.dw
        self.xmax = max([n["x"] for n in self.data["task0"][0]["node"]]) + self.dw
        self.ymin = min([n["y"] for n in self.data["task0"][0]["node"]]) - self.dw
        self.ymax = max([n["y"] for n in self.data["task0"][0]["node"]]) + self.dw
        self.normalize = lambda v, mn, mx: 2 * (v - mn) / (mx - mn) - 1

        nodes = (Node * len(self.data["task0"][0]["node"]))(*[Node(self.normalize(n["x"], self.xmax, self.xmin), self.normalize(n["y"], self.ymax, self.ymin)) for n in self.data["task0"][0]["node"]])
        edges = (Edge * len(self.data["task0"][0]["conn"]))(*[Edge(c["in"], c["out"], c["weight"]) for c in self.data["task0"][0]["conn"]])

        [self.vbo1, self.vbo2, self.ssbo] = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo1)
        glBufferData(GL_ARRAY_BUFFER, 2000, edges, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
        glBufferData(GL_ARRAY_BUFFER, 2000, nodes, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(nodes), nodes, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo)

        [self.vao1, self.vao2] = glGenVertexArrays(2)
        glBindVertexArray(self.vao1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo1)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glVertexAttribIPointer(0, 1, GL_UNSIGNED_INT, sizeof(Edge), GLvoidp(Edge.in_.offset))
        glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, sizeof(Edge), GLvoidp(Edge.out.offset))
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Edge), GLvoidp(Edge.weight.offset))
        glBindVertexArray(0)

        glBindVertexArray(self.vao2)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Node), None)
        glBindVertexArray(0)

        self.node = Shader()
        self.node.attach_shader(node_vert, GL_VERTEX_SHADER)
        self.node.attach_shader(node_frag, GL_FRAGMENT_SHADER)
        self.node.link()

        self.edge = Shader()
        self.edge.attach_shader(edge_vert, GL_VERTEX_SHADER)
        self.edge.attach_shader(edge_geom, GL_GEOMETRY_SHADER)
        self.edge.attach_shader(edge_frag, GL_FRAGMENT_SHADER)
        self.edge.link()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glPointSize(20)
        glClearColor(0, 0, 0, 0)

    def get_pixels(self, task, step):
        nodes = (Node * len(self.data[task][step]["node"]))(*[Node(self.normalize(n["x"], self.xmax, self.xmin), self.normalize(n["y"], self.ymax, self.ymin)) for n in self.data[task][step]["node"]])
        edges = (Edge * len(self.data[task][step]["conn"]))(*[Edge(c["in"], c["out"], c["weight"]) for c in self.data[task][step]["conn"]])

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo1)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(edges), edges)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(nodes), nodes)

        glClear(GL_COLOR_BUFFER_BIT)

        self.edge.use()
        glBindVertexArray(self.vao1)
        glDrawArrays(GL_POINTS, 0, len(self.data[task][step]["conn"]))
        glBindVertexArray(0)
        self.edge.unuse()

        self.node.use()
        glBindVertexArray(self.vao2)
        glDrawArrays(GL_POINTS, 0, len(self.data[task][step]["node"]))
        glBindVertexArray(0)
        self.node.unuse()
        
        pixels = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint32).reshape(*size)

        glfw.swap_buffers(self.window)
        return img


    def __del__(self):
        glDeleteProgram(self.node.handle)
        glDeleteProgram(self.edge.handle)
        glDeleteVertexArrays(2, [self.vao1, self.vao2])
        glDeleteBuffers(3, [self.vbo1, self.vbo2, self.ssbo])

        glfw.destroy_window(self.window)
        glfw.terminate()

class SolverNetwork:
    def __new__(cls, observe: Observe):
        solver = observe.solver()
        network_renderer = GLNetwork(solver)

        ns = [[(c["in"], c["out"]) for c in n["conn"]] for task in solver.values() for n in task]
        nodes = [[(node["x"], node["y"]) for node in n["node"]] for task in solver.values() for n in task]
        xmin, xmax = sys.float_info.max, -sys.float_info.max
        ymin, ymax = sys.float_info.max, -sys.float_info.max
        for network in nodes:
            for x, y in network:
                if xmin > x:
                    xmin = x
                if xmax < x:
                    xmax = x
                if ymin > y:
                    ymin = y
                if ymax < y:
                    ymax = y
        
        n0 = ns[0]
        node0 = nodes[0]

        indices = [(task, i) for task, networks in solver.items() for i in range(len(networks))]

        G = nx.DiGraph()
        G.add_edges_from(n0)
        self = figure(title="solver network", plot_width=500, plot_height=500, output_backend="webgl")
        # n = from_networkx(G, lambda G: { i : node0[i] for i in set([t for e in G.edges() for t in e]) })
        self.x_range = Range1d(xmin - 0.1, xmax + 0.1)
        self.y_range = Range1d(ymin - 0.1, ymax + 0.1)
        img = network_renderer.get_pixels(*indices[0])
        image = self.image_rgba(image=[img], x=xmin, y=ymin, dw=xmax-xmin, dh=ymax-ymin)

        def update_fig(change):
            self.title.text = f"solver network (step={change['new']})"
            image.data_source.data["image"] = [network_renderer.get_pixels(*indices[change["new"]])]
            push_notebook()
            pass

        player_tooltips = MediaPlayer(len(indices), update_fig, lambda: get_screenshot_as_png(self))
        player_tooltips.show()

        return self

class SolverNetworkView:
    def __new__(cls, observe: Observe):
        solver = observe.solver()
        
        has_num_neurocomponents = "num_inputs" in solver["task0"][0]
        if has_num_neurocomponents:
            num_inputs = solver["task0"][0]["num_inputs"]
            num_outputs = solver["task0"][0]["num_outputs"]
        nodes = [[(node["x"], node["y"], node["energy"]) for node in n["nodes"]] if not has_num_neurocomponents else [(node["x"], node["y"], node["energy"]) for node in n["nodes"]][(num_inputs+num_outputs):] for task in solver.values() for n in task]
        indices = [(task, i) for task, networks in solver.items() for i in range(len(networks))]
        px0, py0, energy0 = zip(*nodes[0])
        sgn = lambda v: -1 if v < 0 else 1
        scaler = lambda x: x if -1 <= x <= 1 else sgn(x) * math.log10(abs(x))

        px0 = [scaler(x) for x in px0]
        py0 = [scaler(y) for y in py0]
        energy0 = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in energy0]

        self = figure(title="solver network", plot_width=500, plot_height=500, output_backend="webgl")
        node_data_source = ColumnDataSource(data=dict(px=list(px0), py=list(py0), fill_color=energy0))
        node = Circle(
            x="px", y="py", size=12,
            fill_color="fill_color", line_color="#cc6633", fill_alpha=0.5
        )
        self.add_glyph(node_data_source, node)

        if has_num_neurocomponents:
            input_nodes = [[(node["x"], node["y"], node["energy"]) for node in n["nodes"]][:num_inputs] for task in solver.values() for n in task]
            output_nodes = [[(node["x"], node["y"], node["energy"]) for node in n["nodes"]][num_inputs:num_outputs] for task in solver.values() for n in task]

            ipx0, ipy0, ienergy0 = zip(*input_nodes[0])
            ipx0 = [scaler(x) for x in ipx0]
            ipy0 = [scaler(y) for y in ipy0]
            ienergy0 = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in ienergy0]
            
            opx0, opy0, oenergy0 = zip(*output_nodes[0])
            opx0 = [scaler(x) for x in opx0]
            opy0 = [scaler(y) for y in opy0]
            oenergy0 = [RdBu11[int(10 * e)] if not math.osnan(e) else "#000000" for e in oenergy0]

            input_node_data_source = ColumnDataSource(data=dict(px=list(ipx0), py=list(ipy0), fill_color=ienergy0))
            output_node_data_source = ColumnDataSource(data=dict(px=list(opx0), py=list(opy0), fill_color=oenergy0))

            input_node = Scatter(x="px", y="py", fill_color="fill_color", line_color="#cc6633", fill_alpha=0.5, marker="square")
            output_node = Scatter(x="px", y="py", fill_color="fill_color", line_color="#cc6633", fill_alpha=0.5, marker="inverted_triangle")
            self.add_glyph(input_node_data_source, input_node)
            self.add_glyph(output_node_data_source, output_node)


        conns = [[([scaler(n["nodes"][conn["in"]]["x"]), scaler(n["nodes"][conn["out"]]["x"])], [scaler(n["nodes"][conn["in"]]["y"]), scaler(n["nodes"][conn["out"]]["y"])], conn["weight"]) for conn in n["conns"]] for task in solver.values() for n in task]
        cx0, cy0, ws0 = zip(*conns[0])
        nh = NormalHead(size=12, fill_color="color", line_color="color", fill_alpha=0.5, line_alpha=0.5)
        conn_data_source = ColumnDataSource(data={
            "x_start" : [cx0[i][0] for i in range(len(cx0))],
            "x_end" : [cx0[i][1] for i in range(len(cx0))],
            "y_start" : [cy0[i][0] for i in range(len(cy0))],
            "y_end" : [cy0[i][1] for i in range(len(cy0))],
            "color" : ["blue" if w < 0 else "red" for w in ws0],
            "weight" : [4 * abs(math.tanh(w)) for w in ws0]
        })
        self.add_layout(Arrow(end=nh, x_start="x_start", y_start="y_start", x_end="x_end", y_end="y_end", 
            line_color="color", line_alpha=0.5, line_width="weight",
            source=conn_data_source))

        def update_fig(change):
            self.title.text = f"solver network (step={change['new']})"
            px, py, energy = zip(*nodes[change["new"]])
            px = [scaler(x) for x in px]
            py = [scaler(y) for y in py]
            energy = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in energy]
            node_data_source.data = {
                "px" : px,
                "py" : py,
                "fill_color" : energy
            }
            if len(conns[change["new"]]) == 0:
                conn_data_source.data = {
                    "x_start" : [],
                    "x_end" : [],
                    "y_start" : [],
                    "y_end" : [],
                    "color" : [],
                    "weight" : []
                }
            else:
                cx, cy, ws = zip(*conns[change["new"]])
                conn_data_source.data = {
                    "x_start" : [cx[i][0] for i in range(len(cx))],
                    "x_end" : [cx[i][1] for i in range(len(cx))],
                    "y_start" : [cy[i][0] for i in range(len(cy))],
                    "y_end" : [cy[i][1] for i in range(len(cy))],
                    "color" : ["blue" if w < 0 else "red" for w in ws],
                    "weight" : [4 * abs(math.tanh(w)) for w in ws]
                }
            if has_num_neurocomponents:
                ipx, ipy, ienergy = zip(*input_nodes[change["new"]])
                ipx = [scaler(x) for x in ipx]
                ipy = [scaler(y) for y in ipy]
                ienergy = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in ienergy]
                
                opx, opy, oenergy = zip(*output_nodes[change["new"]])
                opx = [scaler(x) for x in opx]
                opy = [scaler(y) for y in opy]
                oenergy = [RdBu11[int(10 * e)] if not math.osnan(e) else "#000000" for e in oenergy]
                input_node_data_source.data = {
                    "px" : ipx,
                    "py" : ipy,
                    "fill_color" : ienergy
                }
                output_node_data_source.data = {
                    "px" : opx,
                    "py" : opy,
                    "fill_color" : oenergy
                }
            push_notebook()
            pass

        player_tooltips = MediaPlayer(len(indices), update_fig, lambda: get_screenshot_as_png(self))
        player_tooltips.show()

        return self