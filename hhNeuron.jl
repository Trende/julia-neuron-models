module hhNeuron

mutable struct Neuron
    Vref::Float64
    gNa::Float64
    gK::Float64
    gL::Float64
    ENa::Float64
    EK::Float64
    EL::Float64
    Cm::Float64
    v::Float64
    m::Float64
    h::Float64
    n::Float64
end

function advance!(neuron, dt, I)
    # calculate the ion channel currents
    INa = neuron.gNa * neuron.m*neuron.m*neuron.m * neuron.h * (neuron.ENa - neuron.v)
    IK = neuron.gK * neuron.n*neuron.n*neuron.n*neuron.n * (neuron.EK - neuron.v)
    IL = neuron.gL * (neuron.EL - neuron.v)

    # calculate channel kinetics
    aM = 0.1 * (neuron.v - neuron.Vref -25) / ( 1 - exp(-(neuron.v-neuron.Vref-25)/10))
    bM = 4 * exp(-(neuron.v-neuron.Vref)/18)

    aH = 0.07 * exp(-(neuron.v-neuron.Vref)/20)
    bH = 1 / ( 1 + exp( -(neuron.v-neuron.Vref-30)/10 ) )

    aN = 0.01 * (neuron.v-neuron.Vref-10) / ( 1 - exp(-(neuron.v-neuron.Vref-10)/10) )
    bN = 0.125 * exp(-(neuron.v-neuron.Vref)/80)

    # calculate derivatives
    dv_dt = ( INa + IK + IL + I*1e-6 ) / neuron.Cm
    dm_dt = (1.0-neuron.m) * aM - neuron.m * bM
    dh_dt = (1.0-neuron.h) * aH - neuron.h * bH
    dn_dt = (1.0-neuron.n) * aN - neuron.n * bN

    # calculate next step
    neuron.v += dv_dt * dt
    neuron.m += dm_dt * dt
    neuron.h += dh_dt * dt
    neuron.n += dn_dt * dt

    return(neuron)
end

function simulation(dt::Float64=0.01, T::Int=1000)

    const numSteps = T/dt

    # initial membrane potential
    const Vinit = -70.0
    const Vref = -70.0

    # neuron morphology and membrane conductance
    const Smemb = 4000.0                                # surface area
    const Cmemb = 1.0                                   # membrane conductance
    const Cm = Cmemb * Smemb * 1e-8

    # channel parameter
    const gNa = 120.0 * Cmemb * Smemb * 1e-8            # sodium conductance
    const gK = 36.0 * Cmemb * Smemb * 1e-8              # potassium conductance
    const gL = 0.3 * Cmemb * Smemb * 1e-8               # leack conductance
    const ENa = 125.0                                   # sodium equilibrium potential
    const EK = -70.0                                    # potassium equilibrium potential
    const EL = -25.0                                    # leak equilibrium potential

    # initial values for m, h, n
    m = 0.0529
    h = 0.6
    n = 0.32

    v = []
    neuron = Neuron( Vref, gNa, gK, gL, ENa, EK, EL, Cm, Vinit, m, h, n )

    for i = 1:numSteps
        advance!(neuron, dt, 1000)
        push!(v, neuron.v)
    end

    return(v)
end

end
