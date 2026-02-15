import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import casadi
    from casadi import inf
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    import scipy.integrate
    from sympy import (
        Derivative,
        Function,
        atan,
        cos,
        diff,
        init_printing,
        lambdify,
        sin,
        sqrt,
        symbols,
    )
    init_printing()
    return (
        Derivative,
        Function,
        atan,
        casadi,
        cos,
        diff,
        inf,
        lambdify,
        mo,
        np,
        plt,
        scipy,
        sin,
        sqrt,
        symbols,
    )


@app.cell
def _(symbols):
    theta, r, r_pulley, P, theta_max, H = symbols("theta r r_pulley P theta_max H")
    return H, P, r_pulley, theta, theta_max


@app.cell
def _(P, cos, sin, theta):
    # Define parametric equations of winches
    def winch1_with_r(radius):
        return [radius * cos(theta), radius * sin(theta), P * theta]

    def winch2_with_r(radius):
        return [radius * cos(-theta), radius * sin(-theta), P * (-theta)]

    return winch1_with_r, winch2_with_r


@app.cell
def _(P, np, plt, theta, winch1_with_r, winch2_with_r):
    def _eval_curve(curve_expr, theta_values):
        xs, ys, zs = [], [], []
        for theta_val in theta_values:
            subs = {P: 1.0, theta: float(theta_val)}
            xs.append(float(curve_expr[0].evalf(subs=subs)))
            ys.append(float(curve_expr[1].evalf(subs=subs)))
            zs.append(float(curve_expr[2].evalf(subs=subs)))
        return xs, ys, zs

    def plot_winch(radius_expr, theta_now, title, theta_limit=6 * np.pi):
        theta_values_1 = np.arange(0, theta_now, 0.1)
        theta_values_2 = np.arange(0, theta_limit - theta_now, 0.1)

        curve1 = winch1_with_r(radius_expr)
        curve2 = winch2_with_r(radius_expr)
        winch1_evaled = _eval_curve(curve1, theta_values_1)
        winch2_evaled = _eval_curve(curve2, theta_values_2)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.scatter(*winch1_evaled, c=winch1_evaled[2], s=12)
        ax.scatter(*winch2_evaled, c=winch2_evaled[2], s=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return fig

    return (plot_winch,)


@app.cell
def _(mo, np, plot_winch, theta):
    fig_constant = plot_winch(5.0, 2 * np.pi, "Constant Radius")
    fig_variable = plot_winch((1 + theta + 0.1 * theta**2), 3 * np.pi, "Arbitrary Variable Radius")
    mo.md("## Example Winches")
    fig_constant
    fig_variable
    return


@app.cell
def _(Derivative, sqrt, theta):
    def dL_parametric(parametric_eq):
        # Derivative of the length of a parametric equation (integrate to get length)
        return sqrt(
            Derivative(parametric_eq[0], theta) ** 2
            + Derivative(parametric_eq[1], theta) ** 2
            + Derivative(parametric_eq[2], theta) ** 2
        )

    return (dL_parametric,)


@app.cell
def _(Function, dL_parametric, mo, theta, winch1_with_r):
    dL_winch1_general = dL_parametric(winch1_with_r(Function("r")(theta))).doit()
    mo.vstack([mo.md("### dL/dtheta for a generalized winch radius\n"),
    dL_winch1_general])
    return


@app.cell
def _(
    H,
    P,
    atan,
    dL_parametric,
    diff,
    lambdify,
    r_pulley,
    sqrt,
    symbols,
    theta,
    theta_max,
    winch1_with_r,
    winch2_with_r,
):
    # Define expressions for length of middle cables
    def Lmiddle1_with_r(radius):
        return sqrt((P * theta - r_pulley) ** 2 + (H - radius) ** 2 + r_pulley**2)

    def Lmiddle2_with_r(radius):
        return Lmiddle1_with_r(radius).subs(theta, theta_max - theta)

    def Larc1_with_r(radius):
        return r_pulley * atan(P * theta / (H - radius))

    def Larc2_with_r(radius):
        return Larc1_with_r(radius).subs(theta, theta_max - theta)

    # Create a polynomial sequence of desired order
    order = 3
    a = symbols(f"a:{order + 1}")
    R = a[0]
    for i in range(1, order + 1):
        R += a[i] * theta**i

    # Create the full winch length derivative equation
    dLtotalWinch = dL_parametric(winch1_with_r(R)).doit() - dL_parametric(
        winch2_with_r(R)
    ).doit().subs(theta, theta_max - theta)
    dLtotal = (
        dLtotalWinch
        + diff(Lmiddle1_with_r(R), theta)
        + diff(Lmiddle2_with_r(R), theta)
        + diff(Larc1_with_r(R), theta)
        + diff(Larc2_with_r(R), theta)
    )
    R_lam = lambdify([a, theta], R, "sympy")
    return (
        Larc1_with_r,
        Larc2_with_r,
        Lmiddle1_with_r,
        Lmiddle2_with_r,
        R,
        R_lam,
        a,
        dLtotal,
        order,
    )


@app.cell
def _(casadi, lambdify):
    # Modified version from:
    # https://gist.github.com/jgillis/80bb594a6c8fcf55891d1d88b12b68b8
    def sympy2casadi(sympy_expr, sympy_var, casadi_var):
        assert casadi_var.is_vector()
        if casadi_var.shape[1] > 1:
            casadi_var = casadi_var.T
        casadi_var = casadi.vertsplit(casadi_var)
        mapping = {
            "ImmutableDenseMatrix": casadi.blockcat,
            "MutableDenseMatrix": casadi.blockcat,
            "Abs": casadi.fabs,
        }
        func = lambdify(sympy_var, sympy_expr, modules=[mapping, casadi])
        return func(*casadi_var)

    return (sympy2casadi,)


@app.cell
def _(
    H,
    P,
    R,
    a,
    casadi,
    dLtotal,
    inf,
    np,
    order,
    r_pulley,
    sympy2casadi,
    theta,
    theta_max,
):
    # Formulate the final problem in terms of casadi variables
    dtheta = 0.2
    theta_max_num = 15 * np.pi
    P_num = 1.0
    H_num = 40.0
    r_pulley_num = 10.0
    min_radius = 5.0
    max_radius = 8.0

    # somewhat arbitrary starting radius within the allowable range
    a_0 = [(min_radius + max_radius) / 2]
    for _ in range(order):
        a_0.append(0)

    _theta_range = np.arange(0, theta_max_num, dtheta)
    radius_constraint = []
    dLtotal_with_constants = dLtotal.subs({P: P_num, H: H_num, theta_max: theta_max_num, r_pulley: r_pulley_num})

    a_casadi = casadi.SX.sym("a", order + 1)
    theta_casadi = casadi.SX.sym("theta")
    dLtotal_casadi = sympy2casadi(
        dLtotal_with_constants,
        [theta] + list(a),
        casadi.vertcat(theta_casadi, a_casadi),
    )
    R_casadi = sympy2casadi(R, [theta] + list(a), casadi.vertcat(theta_casadi, a_casadi))

    objective_vec = casadi.SX(len(_theta_range), 1)
    for index, theta_value in enumerate(_theta_range):
        objective_vec[index] = casadi.substitute(dLtotal_casadi, theta_casadi, theta_value)
        radius_constraint.append(casadi.substitute(R_casadi, theta_casadi, theta_value))

    objective = casadi.norm_2(objective_vec)
    nlp = {"x": a_casadi, "f": objective, "g": casadi.vertcat(*radius_constraint)}
    solver = casadi.nlpsol("S", "ipopt", nlp)

    res = solver(
        x0=a_0,
        lbg=[min_radius] * len(radius_constraint),
        ubg=[max_radius] * len(radius_constraint),
        lbx=[-inf for _ in a_0],
        ubx=[+inf for _ in a_0],
    )
    res_x = res["x"].full().T[0]
    objective_value = float(res["f"])
    return (
        H_num,
        P_num,
        a_0,
        dtheta,
        objective_value,
        r_pulley_num,
        res_x,
        theta_max_num,
    )


@app.cell
def _(mo, np, objective_value, res_x):
    coeffs = np.array2string(res_x, precision=6, separator=", ")
    mo.md(
        f"""
    ### Optimization Result
    - Objective value: `{objective_value:.6f}`
    - Coefficients: `{coeffs}`
    """
    )
    return


@app.cell
def _(R_lam, dtheta, np, plt, res_x, theta_max_num):
    theta_values = np.arange(0, theta_max_num, dtheta)
    radius_values = np.array([float(R_lam(res_x, theta_value)) for theta_value in theta_values])

    _fig_radius, _ax_radius = plt.subplots(figsize=(7, 4))
    _ax_radius.plot(theta_values, radius_values, linewidth=2)
    _ax_radius.set_title("Optimized Radius Over Theta")
    _ax_radius.set_xlabel("theta")
    _ax_radius.set_ylabel("R(theta)")
    _ax_radius.grid(alpha=0.3)

    _fig_radius
    return


@app.cell
def _(R_lam, np, plot_winch, res_x, theta, theta_max_num):
    fig_optimized_winch = plot_winch(
        R_lam(res_x, theta),
        0.5 * np.pi,
        "Optimized Winch (one side of winch almost fully unwound)",
        theta_limit=theta_max_num,
    )
    fig_optimized_winch
    return


@app.cell
def _(
    H,
    H_num,
    Larc1_with_r,
    Larc2_with_r,
    Lmiddle1_with_r,
    Lmiddle2_with_r,
    P,
    P_num,
    R,
    a,
    a_0,
    dL_parametric,
    lambdify,
    mo,
    np,
    plt,
    r_pulley,
    r_pulley_num,
    res_x,
    scipy,
    theta,
    theta_max,
    theta_max_num,
    winch1_with_r,
    winch2_with_r,
):
    subs = {P: P_num, H: H_num, theta_max: theta_max_num, r_pulley: r_pulley_num}

    dLwinch1 = dL_parametric(winch1_with_r(R)).doit()
    dLwinch2 = dL_parametric(winch2_with_r(R)).doit()
    dLwinch1_lam = lambdify([theta, a], dLwinch1.subs(subs))
    dLwinch2_lam = lambdify([theta, a], dLwinch2.subs(subs))
    Lmiddle1_lam = lambdify([theta, a], Lmiddle1_with_r(R).subs(subs))
    Lmiddle2_lam = lambdify([theta, a], Lmiddle2_with_r(R).subs(subs))
    Larc1_lam = lambdify([theta, a], Larc1_with_r(R).subs(subs))
    Larc2_lam = lambdify([theta, a], Larc2_with_r(R).subs(subs))

    _theta_range_eval = np.arange(0, theta_max_num, 0.2)

    def compute_lengths_res(a_to_use):
        Lwinch1_res = np.array(
            [scipy.integrate.quad(lambda x: dLwinch1_lam(x, a_to_use), 0, theta_val)[0] for theta_val in _theta_range_eval]
        )
        Lwinch2_res = np.array(
            [
                scipy.integrate.quad(lambda x: dLwinch2_lam(x, a_to_use), 0, theta_max_num - theta_val)[0]
                for theta_val in _theta_range_eval
            ]
        )
        Lmiddle1_res = np.array([Lmiddle1_lam(theta_val, a_to_use) for theta_val in _theta_range_eval])
        Lmiddle2_res = np.array([Lmiddle2_lam(theta_val, a_to_use) for theta_val in _theta_range_eval])
        Larc1_res = np.array([Larc1_lam(theta_val, a_to_use) for theta_val in _theta_range_eval])
        Larc2_res = np.array([Larc2_lam(theta_val, a_to_use) for theta_val in _theta_range_eval])
        return np.array([Lmiddle1_res, Lmiddle2_res, Lwinch1_res, Lwinch2_res, Larc1_res, Larc2_res])

    Ltotal_constantradius = np.sum(compute_lengths_res(a_0), axis=0)
    Ltotal_res = np.sum(compute_lengths_res(res_x), axis=0)

    constant_span = float(np.max(Ltotal_constantradius) - np.min(Ltotal_constantradius))
    optimized_span = float(np.max(Ltotal_res) - np.min(Ltotal_res))

    _fig_lengths, _ax_lengths = plt.subplots(figsize=(7, 4))
    _ax_lengths.plot(_theta_range_eval, Ltotal_constantradius, label=f"Constant radius ({constant_span:.4f} span)")
    _ax_lengths.plot(_theta_range_eval, Ltotal_res, label=f"Optimized radius ({optimized_span:.4f} span)")
    _ax_lengths.set_title("Total Cable Length vs Theta")
    _ax_lengths.set_xlabel("theta")
    _ax_lengths.set_ylabel("total length")
    _ax_lengths.grid(alpha=0.3)
    _ax_lengths.legend()

    mo.vstack(
        [mo.md(
            f"""
    ### Length Variation
    - Constant radius coefficients: `{a_0}`
    - Optimized radius coefficients: `{np.array2string(res_x, precision=6, separator=", ")}`
    - Constant radius span: `{constant_span:.6f}`
    - Optimized radius span: `{optimized_span:.6f}`
    """
        )]
    )
    _fig_lengths
    return


if __name__ == "__main__":
    app.run()
