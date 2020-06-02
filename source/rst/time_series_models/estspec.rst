.. _estspec:

.. include:: /_static/includes/header.raw

.. highlight:: julia

******************************************
Estimation of :index:`Spectra`
******************************************

.. index::
    single: Spectra; Estimation

.. contents:: :depth: 2

Overview
============

In a :ref:`previous lecture <arma>` we covered some fundamental properties of covariance stationary linear stochastic processes.

One objective for that lecture was to introduce spectral densities --- a standard and very useful technique for analyzing such processes.

In this lecture we turn to the problem of estimating spectral densities and other related quantities from data.

.. index::
    single: Spectra, Estimation; Fast Fourier Transform

Estimates of the spectral density are computed using what is known as a periodogram --- which in
turn is computed via the famous `fast Fourier transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_.

Once the basic technique has been explained, we will apply it to the analysis of several key macroeconomic time series.

For supplementary reading, see :cite:`Sargent1987` or :cite:`CryerChan2008`.

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics 

.. _periodograms:

:index:`Periodograms`
=====================

:ref:`Recall that <arma_specd>` the spectral density :math:`f` of a covariance stationary process with
autocorrelation function :math:`\gamma` can be written

.. math::

    f(\omega) = \gamma(0) + 2 \sum_{k \geq 1} \gamma(k) \cos(\omega k),
    \qquad \omega \in \mathbb R


Now consider the problem of estimating the spectral density of a given time series, when :math:`\gamma` is unknown.

In particular, let :math:`X_0, \ldots, X_{n-1}` be :math:`n` consecutive observations of a single time series that is assumed to be covariance stationary.

The most common estimator of the spectral density of this process is the *periodogram* of :math:`X_0, \ldots, X_{n-1}`, which is defined as

.. math::
    :label: estspec_p

    I(\omega)
    := \frac{1}{n} \left| \sum_{t=0}^{n-1} X_t e^{i t \omega} \right|^2,
    \qquad \omega \in \mathbb R


(Recall that :math:`|z|` denotes the modulus of complex number :math:`z`)

Alternatively, :math:`I(\omega)` can be expressed as

.. math::

    I(\omega)
    = \frac{1}{n}
    \left\{
    \left[\sum_{t=0}^{n-1} X_t \cos(\omega t) \right]^2
    +
    \left[\sum_{t=0}^{n-1} X_t \sin(\omega t) \right]^2
    \right\}


It is straightforward to show that the function :math:`I` is even and :math:`2
\pi`-periodic (i.e., :math:`I(\omega) = I(-\omega)` and :math:`I(\omega +
2\pi) = I(\omega)` for all :math:`\omega \in \mathbb R`).

From these two results, you will be able to verify that the values of
:math:`I` on :math:`[0, \pi]` determine the values of :math:`I` on all of
:math:`\mathbb R`.

The next section helps to explain the connection between the periodogram and the spectral density.

Interpretation
----------------

.. index::
    single: Periodograms; Interpretation

To interpret the periodogram, it is convenient to focus on its values at the *Fourier frequencies*

.. math::

    \omega_j := \frac{2 \pi j}{n},
    \quad j = 0, \ldots, n - 1


In what sense is :math:`I(\omega_j)` an estimate of :math:`f(\omega_j)`?

The answer is straightforward, although it does involve some algebra.

With a bit of effort one can show that, for any integer :math:`j > 0`,

.. math::

    \sum_{t=0}^{n-1} e^{i t \omega_j }
    = \sum_{t=0}^{n-1} \exp \left\{ i 2 \pi j \frac{t}{n} \right\} = 0


Letting :math:`\bar X` denote the sample mean :math:`n^{-1} \sum_{t=0}^{n-1} X_t`, we then have

.. math::

    n I(\omega_j)
     = \left| \sum_{t=0}^{n-1} (X_t - \bar X) e^{i t \omega_j } \right|^2
     =  \sum_{t=0}^{n-1} (X_t - \bar X) e^{i t \omega_j }
    \sum_{r=0}^{n-1} (X_r - \bar X) e^{-i r \omega_j }


By carefully working through the sums, one can transform this to

.. math::

    n I(\omega_j)
    =  \sum_{t=0}^{n-1} (X_t - \bar X)^2
    + 2 \sum_{k=1}^{n-1} \sum_{t=k}^{n-1} (X_t - \bar X)(X_{t-k} - \bar X)
    \cos(\omega_j k)

Now let

.. math::

    \hat \gamma(k)
    := \frac{1}{n} \sum_{t=k}^{n-1} (X_t - \bar X)(X_{t-k} - \bar X),
    \qquad k = 0,1,\ldots, n-1


This is the sample autocovariance function, the natural "plug-in estimator" of the :ref:`autocovariance function <arma_defs>` :math:`\gamma`.

("Plug-in estimator" is an informal term for an estimator found by replacing expectations with sample means)

With this notation, we can now write

.. math::

    I(\omega_j)
    =  \hat \gamma(0)
    + 2 \sum_{k=1}^{n-1} \hat \gamma(k) \cos(\omega_j k)


Recalling our expression for :math:`f` given :ref:`above <periodograms>`,
we see that :math:`I(\omega_j)` is just a sample analog of :math:`f(\omega_j)`.


Calculation
--------------

.. index::
    single: Periodograms; Computation

Let's now consider how to compute the periodogram as defined in :eq:`estspec_p`.

There are already functions available that will do this for us
--- an example is ``periodogram`` in the ``DSP.jl`` package.

However, it is very simple to replicate their results, and this will give us a platform to make useful extensions.

The most common way to calculate the periodogram is via the discrete Fourier transform,
which in turn is implemented through the `fast Fourier transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_ algorithm.

In general, given a sequence :math:`a_0, \ldots, a_{n-1}`, the discrete
Fourier transform computes the sequence

.. math::

    A_j := \sum_{t=0}^{n-1} a_t \exp \left\{ i 2 \pi \frac{tj}{n} \right\},
    \qquad j = 0, \ldots, n-1

With  :math:`a_0, \ldots, a_{n-1}` stored in Julia array ``a``, the function call ``fft(a)`` returns the values :math:`A_0, \ldots, A_{n-1}` as a Julia array.

It follows that, when the data :math:`X_0, \ldots, X_{n-1}` are stored in array ``X``, the values :math:`I(\omega_j)` at the Fourier frequencies, which are given by

.. math::

    \frac{1}{n} \left| \sum_{t=0}^{n-1} X_t \exp \left\{ i 2 \pi \frac{t j}{n} \right\} \right|^2,
    \qquad j = 0, \ldots, n-1


can be computed by ``abs(fft(X)).^2 / length(X)``.

Note: The Julia function ``abs`` acts elementwise, and correctly handles complex numbers (by computing their modulus, which is exactly what we need).

A function called ``periodogram`` that puts all this together can be found `here <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/estspec.jl>`__.

Let's generate some data for this function using the ``ARMA`` type from `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`__ (see the :ref:`lecture on linear processes <arma>` for more details).

Here's a code snippet that, once the preceding code has been run, generates data from the process

.. math::
    :label: esp_arma

    X_t = 0.5 X_{t-1} + \epsilon_t - 0.8 \epsilon_{t-2}

where :math:`\{ \epsilon_t \}` is white noise with unit variance, and compares the periodogram to the actual spectral density

.. code-block:: julia
  :class: test

  using Test

.. code-block:: julia

    using QuantEcon, Plots, Random
    gr(fmt = :png);
    Random.seed!(42) # For reproducible results.

    n = 40              # Data size
    ϕ = 0.5             # AR parameter
    θ = [0, -0.8]       # MA parameter
    σ = 1.0
    lp = ARMA(ϕ, θ, σ)
    X = simulation(lp, ts_length = n)

    x, y = periodogram(X)
    x_sd, y_sd = spectral_density(lp, two_pi=false, res=120)

    plot(x, y,linecolor="blue", linewidth=2, linealpha=0.5, lab="periodogram")
    plot!(x_sd, y_sd, linecolor="red", linewidth=2, linealpha=0.8, lab="spectral density")

.. code-block:: julia
  :class: test

  @testset begin
    @test y[17] ≈ 0.034025657896215554
    @test x[17] ≈ 2.5132741228718345
    @test y_sd[76] ≈ 1.6587587789586284
  end

This estimate looks rather disappointing, but the data size is only 40, so
perhaps it's not surprising that the estimate is poor.

However, if we try again with ``n = 1200`` the outcome is not much better

.. figure:: /_static/figures/periodogram1.png

The periodogram is far too irregular relative to the underlying spectral density.

This brings us to our next topic.

:index:`Smoothing`
==================

.. index::
    single: Spectra, Estimation; Smoothing

There are two related issues here.

One is that, given the way the fast Fourier transform is implemented, the
number of points :math:`\omega` at which :math:`I(\omega)` is estimated
increases in line with the amount of data.

In other words, although we have more data, we are also using it to estimate more values.

A second issue is that densities of all types are fundamentally hard to
estimate without parametric assumptions.

.. index::
    single: Nonparametric Estimation

Typically, nonparametric estimation of densities requires some degree of smoothing.

The standard way that smoothing is applied to periodograms is by taking local averages.

In other words, the value :math:`I(\omega_j)` is replaced with a weighted
average of the adjacent values

.. math::

    I(\omega_{j-p}), I(\omega_{j-p+1}), \ldots, I(\omega_j), \ldots, I(\omega_{j+p})

This weighted average can be written as

.. math::
    :label: estspec_ws

    I_S(\omega_j) := \sum_{\ell = -p}^{p} w(\ell) I(\omega_{j+\ell})

where the weights :math:`w(-p), \ldots, w(p)` are a sequence of :math:`2p + 1` nonnegative
values summing to one.

In generally, larger values of :math:`p` indicate more smoothing --- more on
this below.

The next figure shows the kind of sequence typically used.

Note the smaller weights towards the edges and larger weights in the center, so that more distant values from :math:`I(\omega_j)` have less weight than closer ones in the sum :eq:`estspec_ws`

.. code-block:: julia

    function hanning_window(M)
        w = [0.5 - 0.5 * cos(2 * pi * n / (M - 1)) for n = 0:(M-1)]
        return w
    end

    window = hanning_window(25) / sum(hanning_window(25))
    x = range(-12, 12, length = 25)
    plot(x, window, color="darkblue", title="Hanning window", ylabel="Weights",
        xlabel="Position in sequence of weights", legend=false, grid=false)

.. code-block:: julia
  :class: test

  @testset begin
    @test window[7] ≈ 0.04166666666666666
    @test window[12] ≈ 0.08191357609537783
  end


Estimation with Smoothing
------------------------------

.. index::
    single: Spectra, Estimation; Smoothing

Our next step is to provide code that will not only estimate the periodogram but also provide smoothing as required.

Such functions have been written in  `estspec.jl <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/estspec.jl>`__ and are available once you've installed `QuantEcon.jl <http://quantecon.org/quantecon-jl>`__.

The `GitHub listing <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/estspec.jl>`__ displays three functions,  ``smooth()``, ``periodogram()``, ``ar_periodogram()``. We will discuss the first two here and the third one :ref:`below <ar_periodograms>`.

The ``periodogram()`` function returns a periodogram, optionally smoothed via the ``smooth()`` function.

Regarding the ``smooth()`` function, since smoothing adds a nontrivial amount of computation, we have applied a fairly terse array-centric method based around ``conv``.

Readers are left either to  explore or simply to use this code according to their interests.

The next three figures each show smoothed and unsmoothed periodograms, as well as the population or "true" spectral density.

(The model is the same as before --- see equation :eq:`esp_arma` --- and there are 400 observations)

From top figure to bottom, the window length is varied from small to large.

.. _fig_window_smoothing:

.. figure:: /_static/figures/window_smoothing.png

In looking at the figure, we can see that for this model and data size, the
window length chosen in the middle figure provides the best fit.

Relative to this value, the first window length provides insufficient
smoothing, while the third gives too much smoothing.

Of course in real estimation problems the true spectral density is not visible
and the choice of appropriate smoothing will have to be made based on
judgement/priors or some other theory.

.. _estspec_pfas:

Pre-Filtering and Smoothing
------------------------------

.. index::
    single: Spectra, Estimation; Pre-Filtering

.. index::
    single: Spectra, Estimation; Smoothing

In the `code listing <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/estspec.jl>`__ we showed three functions from the file ``estspec.jl``.

The third function in the file (``ar_periodogram()``) adds a pre-processing step to periodogram smoothing.

First we describe the basic idea, and after that we give the code.

The essential idea is to

#. Transform the data in order to make estimation of the spectral density more efficient.
#. Compute the periodogram associated with the transformed data.
#. Reverse the effect of the transformation on the periodogram, so that it now
   estimates the spectral density of the original process.

Step 1 is called *pre-filtering* or *pre-whitening*, while step 3 is called *recoloring*.

The first step is called pre-whitening because the
transformation is usually designed to turn the data into something closer to white noise.

Why would this be desirable in terms of spectral density estimation?

The reason is that we are smoothing our estimated periodogram based on
estimated values at nearby points --- recall :eq:`estspec_ws`.

The underlying assumption that makes this a good idea is that the true
spectral density is relatively regular --- the value of :math:`I(\omega)` is close
to that of :math:`I(\omega')` when :math:`\omega` is close to :math:`\omega'`.

This will not be true in all cases, but it is certainly true for white noise.

For white noise, :math:`I` is as regular as possible --- :ref:`it is a constant function <arma_wnsd>`.

In this case, values of :math:`I(\omega')` at points :math:`\omega'` near to :math:`\omega`
provided the maximum possible amount of information about the value :math:`I(\omega)`.

Another way to put this is that if :math:`I` is relatively constant, then we can use a large amount of smoothing without introducing too much bias.

.. _ar_periodograms:

The AR(1) Setting
-------------------

.. index::
    single: Spectra, Estimation; AR(1) Setting

Let's examine this idea more carefully in a particular setting --- where
the data are assumed to be generated by an  AR(1) process.

(More general ARMA settings can be handled using similar techniques to those described below)

Suppose in particular that :math:`\{X_t\}` is covariance stationary and AR(1),
with

.. math::
    :label: estspec_ar_dgp

    X_{t+1} = \mu + \phi X_t + \epsilon_{t+1}

where :math:`\mu` and :math:`\phi \in (-1, 1)` are unknown parameters and :math:`\{ \epsilon_t \}` is white noise.

It follows that if we regress :math:`X_{t+1}` on :math:`X_t` and an intercept, the residuals
will approximate white noise.

Let

* :math:`g` be the spectral density of :math:`\{ \epsilon_t \}` --- a constant function, as discussed above

* :math:`I_0` be the periodogram estimated from the residuals --- an estimate of :math:`g`

* :math:`f` be the spectral density of :math:`\{ X_t \}` --- the object we are trying to estimate

In view of :ref:`an earlier result <arma_spec_den>` we obtained while discussing ARMA processes, :math:`f` and :math:`g` are related by

.. math::
    :label: ar_sdsc

    f(\omega) = \left| \frac{1}{1 - \phi e^{i\omega}} \right|^2 g(\omega)

This suggests that the recoloring step, which constructs an estimate :math:`I` of :math:`f` from :math:`I_0`, should set

.. math::

    I(\omega) = \left| \frac{1}{1 - \hat \phi e^{i\omega}} \right|^2 I_0(\omega)

where :math:`\hat \phi` is the OLS estimate of :math:`\phi`.

The code for ``ar_periodogram()`` --- the third function in ``estspec.jl`` --- does exactly this. (See the code `here <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/estspec.jl>`__).

The next figure shows realizations of the two kinds of smoothed periodograms

#. "standard smoothed periodogram", the ordinary smoothed periodogram, and
#. "AR smoothed periodogram", the pre-whitened and recolored one generated by ``ar_periodogram()``

The periodograms are calculated from time series drawn from :eq:`estspec_ar_dgp` with :math:`\mu = 0` and :math:`\phi = -0.9`.

Each time series is of length 150.

The difference between the three subfigures is just randomness --- each one uses a different draw of the time series.

.. _fig_ar_smoothed_periodogram:

.. figure:: /_static/figures/ar_smoothed_periodogram.png

In all cases, periodograms are fit with the "hamming" window and window length of 65.

Overall, the fit of the AR smoothed periodogram is much better, in the sense
of being closer to the true spectral density.

Exercises
=============

.. _estspec_ex1:

Exercise 1
------------

Replicate :ref:`this figure <fig_window_smoothing>` (modulo randomness).

The model is as in equation :eq:`esp_arma` and there are 400 observations.

For the smoothed periodogram, the window type is "hamming".

.. _estspec_ex2:

Exercise 2
------------

Replicate :ref:`this figure <fig_ar_smoothed_periodogram>` (modulo randomness).

The model is as in equation :eq:`estspec_ar_dgp`, with :math:`\mu = 0`, :math:`\phi = -0.9`
and 150 observations in each time series.

All periodograms are fit with the "hamming" window and window length of 65.

.. The exercise 2 and 3 are not solvable in Julia because function `ar_periodgram` is broken.

Solutions
============

Exercise 1
----------

.. code-block:: julia
    :class: test

    using Random
    Random.seed!(42);  # reproducible results

.. code-block:: julia

    n = 400
    ϕ = 0.5
    θ = [0, -0.8]
    σ = 1.0
    lp = ARMA(ϕ, θ, 1.0)
    X = simulation(lp, ts_length = n)

    xs = []
    x_sds = []
    x_sms = []
    ys = []
    y_sds = []
    y_sms = []
    titles = []

    for (i, wl) in enumerate([15, 55, 175])  # window lengths
        x, y = periodogram(X)
        push!(xs, x)
        push!(ys, y)

        x_sd, y_sd = spectral_density(lp, two_pi=false, res=120)
        push!(x_sds, x_sd)
        push!(y_sds, y_sd)

        x, y_smoothed = periodogram(X, "hamming", wl)
        push!(x_sms, x)
        push!(y_sms, y_smoothed)

        t = "window length = $wl"
        push!(titles, t)
    end

.. code-block:: julia
  :class: test

  @testset begin
    @test y_sds[2][12] ≈ 1.0359408815913638
    @test y_sms[3][45] ≈ 2.7396611185705604
    @test ys[1][50] ≈ 4.245609056262289
  end

.. code-block:: julia

    plot(xs, ys, layout=(3,1), color=:blue, alpha=0.5,
        linewidth=2, label=["periodogram" "" ""])
    plot!(x_sds, y_sds, layout=(3,1), color=:red, alpha=0.8,
        linewidth=2, label=["spectral density" "" ""])
    plot!(x_sms, y_sms, layout=(3,1), color=:black,
        linewidth=2, label=["smoothed periodogram" "" ""])
    plot!(title=reshape(titles,1,length(titles)))

Exercise 2
----------
.. code-block:: julia
    :class: test

    Random.seed!(42);  # reproducible results

.. code-block:: julia

    lp2 = ARMA(-0.9, 0.0, 1.0)
    wl = 65
    p = plot(layout=(3,1))

    for i in 1:3
        X = simulation(lp2,ts_length=150)
        plot!(p[i],xlims = (0,pi))

        x_sd, y_sd = spectral_density(lp2,two_pi=false, res=180)
        plot!(p[i],x_sd, y_sd, linecolor=:red, linestyle=:solid,
            yscale=:log10, linewidth=2, linealpha=0.75,
            label="spectral density",legend=:topleft)

        x, y_smoothed = periodogram(X, "hamming", wl)
        plot!(p[i],x, y_smoothed, linecolor=:black, linestyle=:solid,
            yscale=:log10, linewidth=2, linealpha=0.75,
            label="standard smoothed periodogram",legend=:topleft)

        x, y_ar = ar_periodogram(X, "hamming", wl)
        plot!(p[i],x, y_ar, linecolor=:blue, linestyle=:solid,
            yscale=:log10, linewidth=2, linealpha=0.75,
            label="AR smoothed periodogram",legend=:topleft)
    end
    p

.. code-block:: julia
  :class: test

  Random.seed!(42)
  y_sd = spectral_density(lp2,two_pi=false, res=180)

  @testset begin
    @test y_sd[2][100] ≈ 0.6616951403067453
    @test y_sd[1][100] ≈ 1.7375288977954721
  end
