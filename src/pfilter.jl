using Plots
using LinearAlgebra
using DelimitedFiles
using Compat, Random, Distributions, StatsBase
Random.seed!(21);

mutable struct ParticleGroup
    locations;
    weights;
end

struct NoiseDists
    actual;
    estimated;
    perturb;
end

function elevAboveSeaLevel(x)
    sin_term = sin(0.01 * x + 3);
    cos_term = cos((0.01 * x - 2) * (0.005 * x - 5));
    prefix = (100000 * x) / (1 + (x + 500) ^ 2);
    elev = prefix * (cos_term - sin_term);
    return elev;
end

function get_measurement(x, distribution)
    msmt = elevAboveSeaLevel(x);
    msmt += rand(distribution);
    return msmt;
end

function get_likelihood(x, msmt, distribution)
    ytrue = elevAboveSeaLevel(x);
    residual = msmt - ytrue;
    prob = pdf(distribution, residual);
    return prob;
end

function init_particles(x0, x1, Np)
    scale = rand(Uniform(), Np);
    locations = x0 .+ (x1 - x0) .* scale;
    return ParticleGroup(locations, ones(Np));
end

function init_histories(Nt, Np)
    xtrue = zeros(Nt);
    x = zeros(Nt, Np);
    return (xtrue, x);
end

function update_weights!(particles, xtrue, noise)
    msmt = get_measurement(xtrue, noise.actual);
    particles.weights = get_likelihood.(
        particles.locations, 
        msmt, 
        noise.estimated
    );
end

function update_locations!(particles)
    wsample!(
        particles.locations, 
        particles.weights, 
        particles.locations, 
        replace=true
    );
end

function move!(particles, noise, dt, velocity)
    particles.locations .+= dt * velocity;
    particles.locations .+= rand.(noise.perturb);
end

function forward!(particles, xtrue, noise, dt, velocity)
    update_weights!(particles, xtrue, noise);
    update_locations!(particles);
    move!(particles, noise, dt, velocity);
end

function run_filter(x0, x1, xtrue, Np, Nt, dt, velocity, σ_true, σ_est, σ_pert)
    particles = init_particles(x0, x1, Np);
    xtrue_history, x_history = init_histories(Nt, Np);
    noise = NoiseDists(map((σ) -> Normal(0, σ), [σ_true, σ_est, σ_pert])...);
    for t = 1 : Nt
        forward!(particles, xtrue, noise, dt, velocity);
        xtrue += dt * velocity;
        xtrue_history[t] = xtrue;
        x_history[t, :] = particles.locations;
    end 
    return (xtrue_history, x_history);
end

function make_marker(fpath, scale=5)
    verts = scale .* readdlm(fpath); 
    x, y = verts[:, 1], verts[:, 2];
    return [Shape(x, y)];
end

function make_mountain(x0, x1)
    x = collect(x0 : 1 : x1);
    y = elevAboveSeaLevel.(x);
    return (x, y);
end

function make_animation(x, y, x0, y0, x1, y1, x_history, xtrue_history, aeroplane, Nt)
    anim = @animate for t = 1 : Nt
        xest_t = x_history[t, :];
        yest_t = elevAboveSeaLevel.(xest_t);
        plot(x, y, xlims=(x0, x1), ylims=(y0, y1), legend=false);
        scatter!(xest_t, yest_t, markersize=1, color="darkblue");
        scatter!([xtrue_history[t] - 50], [y1 - 50], shape=aeroplane, ms=20);
    end
    return anim;
end

function main()
    Nt, Np = 200, 5000;
    dt, velocity = 0.01, 200;
    x0, x1, xtrue = 0, 1000, 100;
    σ_true, σ_est, σ_pert  = 50, 40, 0.1 * dt * velocity;

    args = (x0, x1, xtrue, Np, Nt, dt, velocity, σ_true, σ_est, σ_pert)
    xtrue_history, x_history = run_filter(args...);

    x, y = make_mountain(x0, x1);
    y0, y1 = quantile(y[:], [0, 1]) .+ [-10, +75];
    aeroplane = make_marker("../data/airplane_verts.txt")

    anim = make_animation(x, y, x0, y0, x1, y1, x_history, xtrue_history, aeroplane, Nt);
    gif(anim, "../data/demo.gif", fps=20);
end

main();
