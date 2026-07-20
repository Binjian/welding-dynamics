# -*- coding: utf-8 -*-
"""PyVista 'html' 后端内联渲染工具 (源自 robot6_weave_interactive_demo).

- :func:`html_view` — 导出自包含 vtk.js 场景并**复原内核 std 流**
  (VTK + IPython>=9 的"下一格永远运行中"卡死修复), 可选注入场景内
  客户端图层切换复选框 (切换时**视角保持不变**)。
- :func:`add_mouse_hint` — 鼠标操作角标。
- :data:`TAG_ON` / :data:`TAG_OFF` — 图层组透明度指纹, 加到基准透明度上
  (万分位, 视觉不可分辨): 勾选时显示 ON 组、隐藏 OFF 组, 反之亦然。
"""
import sys

from IPython.display import HTML

# 导入时的原始内核流 (在任何 VTK 渲染污染之前 import 本模块)
_STD_STREAMS = (sys.stdout, sys.stderr)

# 图层组指纹: opacity*10000 % 10 == 1 -> ON 组 (勾选显示), == 2 -> OFF 组
TAG_ON, TAG_OFF = 1e-4, 2e-4

# 注入的场景内切换脚本: 只用单引号 (srcdoc 属性里双引号已转义为 &quot;)
_TOGGLE_JS = """
(function(){
  var tries = 0;
  function grp(a){var op=a.getProperty().getOpacity();
    var f=Math.round(op*10000)%10; return f===1?1:(f===2?2:0);}
  function apply(showOn){
    var g=window.global; if(!g||!g.renderWindow) return false;
    var rr=g.renderWindow.getRenderers(); var found=false;
    for(var j=0;j<rr.length;j++){          // 场景 actor 在同步器新建的
      var acts=rr[j].getActors();          // renderer 里 (不是 rr[0]), 全扫
      for(var i=0;i<acts.length;i++){var k=grp(acts[i]);
        if(k===1){acts[i].setVisibility(showOn);found=true;}
        if(k===2){acts[i].setVisibility(!showOn);found=true;}}}
    if(found) g.renderWindow.render();
    return found;
  }
  function init(){
    if(!apply(__INIT__)){ if(tries++ < 40) setTimeout(init,150); return; }
    var d=document.createElement('div');
    d.style.cssText='position:absolute;top:6px;left:50%;'+
      'transform:translateX(-50%);z-index:10;background:rgba(255,255,255,.88);'+
      'padding:3px 8px;border-radius:4px;font:12px sans-serif;color:#333;';
    var c=document.createElement('input'); c.type='checkbox';
    c.checked=__INIT__; c.onchange=function(){apply(c.checked);};
    var l=document.createElement('label'); l.style.cursor='pointer';
    l.appendChild(c); l.appendChild(document.createTextNode(' __LABEL__'));
    d.appendChild(l); document.body.appendChild(d);
  }
  setTimeout(init,150);
})();"""

_ANCHOR = "OfflineLocalView.load(container, { base64Str });"


def inject_layer_toggle(html, label, initial):
    """向导出 html (iframe srcdoc) 注入场景内图层切换复选框脚本。

    带 :data:`TAG_ON` / :data:`TAG_OFF` 指纹的 actor 分别在勾选/取消时
    显示; 切换只在浏览器端翻转可见性并重渲染 —— **相机完全不动**。
    两组 actor 都要以可见状态加入场景 (隐藏 actor 不会被导出), 加载后
    由脚本立即按 ``initial`` 设置可见性。
    """
    js = (_TOGGLE_JS.replace('__INIT__', 'true' if initial else 'false')
                    .replace('__LABEL__', label))
    assert html.count(_ANCHOR) == 1, "html 结构变化: 未找到注入锚点"
    return html.replace(_ANCHOR, _ANCHOR + js)


def html_view(p, toggle=None):
    """导出自包含 vtk.js 场景 ('html' 后端, 无 trame 服务器, 取 iframe 以
    text/html 内联, VS Code / JupyterLab 均可渲染) 并**复原内核 std 流**:
    Linux 下 add_text 触发 VTK 的 matplotlib-mathtext 探测
    (vtkPythonInterpreter), 后者把 sys.stdout/stderr 换成**只读**的
    vtkPythonStdStreamCaptureHelper; IPython>=9 每格 run_cell 的 _tee 要对
    stream.write 赋值, 遇只读对象抛 AttributeError, 内核 execute_request
    中断且不回复 —— 表现为**下一格永远"运行中"**。渲染后换回原对象即可。

    ``toggle=(label, initial)``: 注入场景内复选框, 客户端切换指纹标记的
    两组 actor (见 :func:`inject_layer_toggle`), **视角保持不变**。
    """
    try:
        viewer = p.show(jupyter_backend='html', return_viewer=True)
    finally:
        for name, orig in zip(("stdout", "stderr"), _STD_STREAMS):
            if type(getattr(sys, name)).__name__ == "vtkPythonStdStreamCaptureHelper":
                setattr(sys, name, orig)
    if not getattr(viewer, 'value', None):
        return viewer
    body = viewer.value
    if toggle is not None:
        body = inject_layer_toggle(body, *toggle)
    # <div> 包裹自包含 iframe: 否则 IPython 的 HTML 把裸 <iframe>…</iframe>
    # 误判为应改用 IFrame, 每次渲染刷一条 UserWarning (display.py:444)。
    return HTML(f'<div>{body}</div>')


def add_mouse_hint(p):
    """鼠标操作角标。html 场景的 vtk.js trackball 交互全挂在左键+修饰键:
    左键拖拽=旋转, Shift+左键=平移(即时, 无需重绘), Ctrl/Alt+左键=自旋,
    滚轮=缩放。VTK 默认字体无中文字形, 提示用英文。注意: 自包含 html
    场景没有回传通道 (这正是它在 VS Code webview 里能用的原因), 鼠标
    取景只活在浏览器端 —— 服务端滑块才是可持久化/可复现的取景。"""
    p.add_text("drag: rotate | Shift+drag: pan | scroll: zoom",
               position='upper_right', font_size=8, color="#888888")
