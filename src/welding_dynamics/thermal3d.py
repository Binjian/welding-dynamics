# -*- coding: utf-8 -*-
"""模块 9: GoldakFDM 三维场的 Mayavi 体渲染 与 OpenFOAM 算例导出

- OpenFOAMExporter: 把 GoldakFDM 的结构化网格 + 温度场写成完整 OpenFOAM 算例
  (constant/polyMesh 六面体网格 + 0/ 与时间目录下的 T/Tpeak volScalarField)。
  生成 case.foam 占位文件, 可在 ParaView 中直接打开。半对称用 symmetryPlane patch。
- render: Mayavi (mlab) 体渲染, 将半模型沿 y=0 镜像为全熔池, 画熔合线/HAZ 等温面。
  mayavi 为可选依赖, 延迟导入 (uv sync --extra viz)。
"""
from pathlib import Path
import numpy as np


# ----------------------------------------------------------------------------
# OpenFOAM 导出 (纯 numpy, 无需 mayavi)
# ----------------------------------------------------------------------------
def _foam_header(cls, obj, loc, note=None):
    note_line = f'    note        "{note}";\n' if note else ""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| welding-dynamics : GoldakFDM -> OpenFOAM export                            |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f'    location    "{loc}";\n'
        + note_line +
        f"    object      {obj};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


class OpenFOAMExporter:
    """把一个已求解的 GoldakFDM 实例导出为 OpenFOAM 算例目录。

    单元 (i,j,k) 线性序   c = i + Nx*(j + Ny*k)   <-> field.ravel(order='F')
    顶点 (i,j,k) 线性序   p = i + (Nx+1)*(j + (Ny+1)*k)
    patch: symmetryPlane (y=0 对称面), top (z=0 工件上表面), farField (其余远场)。
    """

    def __init__(self, fdm):
        self.fdm = fdm
        self.Nx, self.Ny, self.Nz = fdm.Nx, fdm.Ny, fdm.Nz
        self.dx = fdm.dx

    # -- 几何/拓扑 ----------------------------------------------------------
    def _pid(self, i, j, k):
        npx, npy = self.Nx + 1, self.Ny + 1
        return i + npx * (j + npy * k)

    def _cid(self, i, j, k):
        return i + self.Nx * (j + self.Ny * k)

    def _points(self):
        npx, npy, npz = self.Nx + 1, self.Ny + 1, self.Nz + 1
        ii = np.arange(npx) * self.dx
        jj = np.arange(npy) * self.dx
        kk = np.arange(npz) * self.dx
        Xp, Yp, Zp = np.meshgrid(ii, jj, kk, indexing="ij")
        return np.column_stack([Xp.ravel("F"), Yp.ravel("F"), Zp.ravel("F")])

    def _build_faces(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        pid, cid = self._pid, self._cid

        def grid(ri, rj, rk):
            return np.meshgrid(ri, rj, rk, indexing="ij")

        # --- 内部面 (owner=较小单元, 法向 owner->neighbour 为 +轴向) ---
        # x 法向面, i=1..Nx-1
        I, J, K = grid(np.arange(1, Nx), np.arange(Ny), np.arange(Nz))
        xo, xn = cid(I - 1, J, K), cid(I, J, K)
        xq = np.stack([pid(I, J, K), pid(I, J + 1, K),
                       pid(I, J + 1, K + 1), pid(I, J, K + 1)], -1)
        # y 法向面, j=1..Ny-1
        I, J, K = grid(np.arange(Nx), np.arange(1, Ny), np.arange(Nz))
        yo, yn = cid(I, J - 1, K), cid(I, J, K)
        yq = np.stack([pid(I, J, K), pid(I, J, K + 1),
                       pid(I + 1, J, K + 1), pid(I + 1, J, K)], -1)
        # z 法向面, k=1..Nz-1
        I, J, K = grid(np.arange(Nx), np.arange(Ny), np.arange(1, Nz))
        zo, zn = cid(I, J, K - 1), cid(I, J, K)
        zq = np.stack([pid(I, J, K), pid(I + 1, J, K),
                       pid(I + 1, J + 1, K), pid(I, J + 1, K)], -1)

        owner = np.concatenate([a.ravel() for a in (xo, yo, zo)])
        neigh = np.concatenate([a.ravel() for a in (xn, yn, zn)])
        quads = np.concatenate([q.reshape(-1, 4) for q in (xq, yq, zq)])
        # 上三角排序: 先按 owner 再按 neighbour
        order = np.lexsort((neigh, owner))
        owner, neigh, quads = owner[order], neigh[order], quads[order]
        n_internal = owner.size

        # --- 边界面 (法向朝外) ---
        bnd_owner, bnd_quads, patches = [], [], []

        def add_patch(name, ptype, o, q):
            patches.append((name, ptype, len(np.concatenate(bnd_owner))
                            if bnd_owner else 0, o.size))
            bnd_owner.append(o)
            bnd_quads.append(q)

        # symmetryPlane: y=0 (j=0), 外法向 -y
        I, J, K = grid(np.arange(Nx), [0], np.arange(Nz))
        o = cid(I, J, K).ravel()
        q = np.stack([pid(I, 0, K), pid(I + 1, 0, K),
                      pid(I + 1, 0, K + 1), pid(I, 0, K + 1)], -1).reshape(-1, 4)
        add_patch("symmetryPlane", "symmetryPlane", o, q)

        # top: z=0 (k=0, 工件上表面/焊枪侧), 外法向 -z
        I, J, K = grid(np.arange(Nx), np.arange(Ny), [0])
        o = cid(I, J, K).ravel()
        q = np.stack([pid(I, J, 0), pid(I, J + 1, 0),
                      pid(I + 1, J + 1, 0), pid(I + 1, J, 0)], -1).reshape(-1, 4)
        add_patch("top", "patch", o, q)

        # farField: xMin(-x) xMax(+x) yMax(+y) zMax(+z) 合并
        far_o, far_q = [], []
        # xMin
        I, J, K = grid([0], np.arange(Ny), np.arange(Nz))
        far_o.append(cid(I, J, K).ravel())
        far_q.append(np.stack([pid(0, J, K), pid(0, J, K + 1),
                               pid(0, J + 1, K + 1), pid(0, J + 1, K)], -1).reshape(-1, 4))
        # xMax
        I, J, K = grid([Nx - 1], np.arange(Ny), np.arange(Nz))
        far_o.append(cid(I, J, K).ravel())
        far_q.append(np.stack([pid(Nx, J, K), pid(Nx, J + 1, K),
                               pid(Nx, J + 1, K + 1), pid(Nx, J, K + 1)], -1).reshape(-1, 4))
        # yMax
        I, J, K = grid(np.arange(Nx), [Ny - 1], np.arange(Nz))
        far_o.append(cid(I, J, K).ravel())
        far_q.append(np.stack([pid(I, Ny, K), pid(I, Ny, K + 1),
                               pid(I + 1, Ny, K + 1), pid(I + 1, Ny, K)], -1).reshape(-1, 4))
        # zMax
        I, J, K = grid(np.arange(Nx), np.arange(Ny), [Nz - 1])
        far_o.append(cid(I, J, K).ravel())
        far_q.append(np.stack([pid(I, J, Nz), pid(I + 1, J, Nz),
                               pid(I + 1, J + 1, Nz), pid(I, J + 1, Nz)], -1).reshape(-1, 4))
        add_patch("farField", "patch", np.concatenate(far_o),
                  np.concatenate(far_q))

        owner = np.concatenate([owner] + bnd_owner)
        quads = np.concatenate([quads] + bnd_quads)
        # 修正 patch 的 startFace 为全局偏移 (内部面之后)
        patches = [(n, t, n_internal + s, c) for (n, t, s, c) in patches]
        return quads, owner, neigh, n_internal, patches

    # -- 文件写出 -----------------------------------------------------------
    @staticmethod
    def _write_list(fh, header, rows):
        fh.write(header)
        fh.write(f"{len(rows)}\n(\n")
        fh.write("\n".join(rows))
        fh.write("\n)\n")

    def _write_mesh(self, mesh_dir, points, quads, owner, neigh, patches):
        mesh_dir.mkdir(parents=True, exist_ok=True)
        note = (f"nPoints:{len(points)} nCells:{self.Nx*self.Ny*self.Nz} "
                f"nFaces:{len(quads)} nInternalFaces:{len(neigh)}")
        with open(mesh_dir / "points", "w") as f:
            self._write_list(f, _foam_header("vectorField", "points", "constant/polyMesh"),
                             [f"({p[0]:.6g} {p[1]:.6g} {p[2]:.6g})" for p in points])
        with open(mesh_dir / "faces", "w") as f:
            self._write_list(f, _foam_header("faceList", "faces", "constant/polyMesh"),
                             [f"4({q[0]} {q[1]} {q[2]} {q[3]})" for q in quads])
        with open(mesh_dir / "owner", "w") as f:
            self._write_list(f, _foam_header("labelList", "owner", "constant/polyMesh", note),
                             [str(int(o)) for o in owner])
        with open(mesh_dir / "neighbour", "w") as f:
            self._write_list(f, _foam_header("labelList", "neighbour", "constant/polyMesh", note),
                             [str(int(n)) for n in neigh])
        with open(mesh_dir / "boundary", "w") as f:
            f.write(_foam_header("polyBoundaryMesh", "boundary", "constant/polyMesh"))
            f.write(f"{len(patches)}\n(\n")
            for name, ptype, start, count in patches:
                f.write(f"    {name}\n    {{\n        type        {ptype};\n"
                        f"        nFaces      {count};\n"
                        f"        startFace   {start};\n    }}\n")
            f.write(")\n")

    def _write_field(self, path, obj, time, values, T0, uniform=False):
        bnd = (
            "boundaryField\n{\n"
            "    symmetryPlane { type symmetryPlane; }\n"
            "    top           { type zeroGradient; }\n"
            f"    farField      {{ type fixedValue; value uniform {T0:.6g}; }}\n"
            "}\n"
        )
        with open(path, "w") as f:
            f.write(_foam_header("volScalarField", obj, str(time)))
            f.write("dimensions      [0 0 0 1 0 0 0];\n\n")
            if uniform:
                f.write(f"internalField   uniform {T0:.6g};\n\n")
            else:
                f.write("internalField   nonuniform List<scalar>\n")
                f.write(f"{values.size}\n(\n")
                f.write("\n".join(f"{v:.6g}" for v in values))
                f.write("\n)\n;\n\n")
            f.write(bnd)

    def export(self, case_dir, t_end=5.0):
        """写出完整算例。返回算例目录 Path。"""
        fdm = self.fdm
        case = Path(case_dir)
        T0 = fdm.T0
        tname = f"{t_end:g}"

        points = self._points()
        quads, owner, neigh, _, patches = self._build_faces()
        self._write_mesh(case / "constant" / "polyMesh",
                         points, quads, owner, neigh, patches)

        # 0/ 初始场 (均匀 T0)
        (case / "0").mkdir(parents=True, exist_ok=True)
        self._write_field(case / "0" / "T", "T", 0, None, T0, uniform=True)

        # 末时刻: 最终温度场 + 峰值温度场
        (case / tname).mkdir(parents=True, exist_ok=True)
        self._write_field(case / tname / "T", "T", tname,
                          fdm.T.ravel(order="F"), T0)
        if hasattr(fdm, "peak"):
            self._write_field(case / tname / "Tpeak", "Tpeak", tname,
                              fdm.peak.ravel(order="F"), T0)

        self._write_system_constant(case, t_end)
        (case / "case.foam").touch()      # ParaView 占位文件
        return case

    def _write_system_constant(self, case, t_end):
        """最小可运行 laplacianFoam 配置 (导出物理 DT=alpha), 便于复算/可视化。"""
        sysdir = case / "system"
        sysdir.mkdir(parents=True, exist_ok=True)
        dt = 0.4 * self.dx ** 2 / (6 * self.fdm.alpha)
        with open(sysdir / "controlDict", "w") as f:
            f.write(_foam_header("dictionary", "controlDict", "system"))
            f.write("application     laplacianFoam;\n"
                    "startFrom       startTime;\nstartTime       0;\n"
                    "stopAt          endTime;\n"
                    f"endTime         {t_end:g};\n"
                    f"deltaT          {dt:.4g};\n"
                    "writeControl    runTime;\n"
                    f"writeInterval   {t_end:g};\n"
                    "runTimeModifiable true;\n")
        with open(sysdir / "fvSchemes", "w") as f:
            f.write(_foam_header("dictionary", "fvSchemes", "system"))
            f.write("ddtSchemes      { default Euler; }\n"
                    "gradSchemes     { default Gauss linear; }\n"
                    "laplacianSchemes{ default Gauss linear corrected; }\n"
                    "divSchemes      { default none; }\n"
                    "interpolationSchemes { default linear; }\n"
                    "snGradSchemes   { default corrected; }\n")
        with open(sysdir / "fvSolution", "w") as f:
            f.write(_foam_header("dictionary", "fvSolution", "system"))
            f.write("solvers { T { solver PCG; preconditioner DIC; "
                    "tolerance 1e-06; relTol 0; } }\n"
                    "SIMPLE { }\n")
        with open(case / "constant" / "transportProperties", "w") as f:
            f.write(_foam_header("dictionary", "transportProperties", "constant"))
            f.write(f"DT              DT [0 2 -1 0 0 0 0] {self.fdm.alpha:.6g};\n")


def export_openfoam(fdm, case_dir="results/openfoam_case", t_end=5.0):
    """便捷函数: 导出 GoldakFDM 到 OpenFOAM 算例。"""
    return OpenFOAMExporter(fdm).export(case_dir, t_end=t_end)


# ----------------------------------------------------------------------------
# Mayavi 体渲染 (可选依赖, 延迟导入)
# ----------------------------------------------------------------------------
def render(fdm, field="peak", outfile=None, offscreen=False,
           size=(1000, 700)):
    """用 Mayavi 渲染熔池/HAZ 等温面。

    field: "peak" 峰值温度场(熔合区+HAZ) 或 "final" 末时刻温度场。
    offscreen=True 时离屏渲染并保存 outfile (需要 VTK 离屏支持/xvfb)。
    需要可选依赖 mayavi: `uv sync --extra viz`。
    """
    try:
        from mayavi import mlab
    except ImportError as e:
        raise ImportError(
            "render() 需要可选依赖 mayavi。请运行 `uv sync --extra viz` "
            "或 `uv pip install mayavi`。"
        ) from e

    if offscreen:
        mlab.options.offscreen = True

    T = fdm.peak if field == "peak" else fdm.T
    Tfull = np.concatenate([T[:, :0:-1, :], T], axis=1)
    x = fdm.x * 1e3
    y = np.concatenate([-fdm.y[:0:-1], fdm.y]) * 1e3
    z = -fdm.z * 1e3
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    Tm, T0 = fdm.Tm, fdm.T0
    haz = 1073.0 if Tm > 1073.0 else 0.5 * (Tm + T0)   # ~800°C HAZ 线

    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=size)
    src = mlab.pipeline.scalar_field(X, Y, Z, Tfull)
    # 熔合区等温面 (熔点) — 不透明红
    mlab.pipeline.iso_surface(src, contours=[float(Tm)], color=(0.85, 0.1, 0.1),
                              opacity=0.55)
    # HAZ 等温面 — 半透明橙
    mlab.pipeline.iso_surface(src, contours=[float(haz)], color=(1.0, 0.6, 0.1),
                              opacity=0.25)
    # 对称面温度切片
    mlab.pipeline.image_plane_widget(src, plane_orientation="y_axes",
                                     slice_index=Tfull.shape[1] // 2,
                                     colormap="jet")
    mlab.colorbar(title="T [K]", orientation="vertical", nb_labels=6)
    mlab.axes(xlabel="x [mm]", ylabel="y [mm]", zlabel="depth [mm]",
              ranges=[x.min(), x.max(), y.min(), y.max(), z.min(), z.max()])
    mlab.title(f"GoldakFDM 3D ({field}): melt {Tm:.0f} K / HAZ {haz:.0f} K",
               size=0.4, height=0.92)
    mlab.view(azimuth=-60, elevation=65, distance="auto")

    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        mlab.savefig(str(outfile), magnification=1)
    if not offscreen:
        mlab.show()
    else:
        mlab.close(fig)
    return outfile
