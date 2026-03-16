"""
    keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)

Convert Keplerian orbital elements to Cartesian state vector [x, y, z, vx, vy, vz].
Pure-math implementation (no external dependencies).

# Arguments
- `a`: Semi-major axis [m]
- `e`: Eccentricity [-]
- `inc`: Inclination [rad]
- `RAAN`: Right Ascension of Ascending Node [rad]  
- `argp`: Argument of perigee [rad]
- `true_anom`: True anomaly [rad]
- `mu`: Gravitational parameter [m³/s²]

# Returns
- `Vector{Float64}` of length 6: [x, y, z, vx, vy, vz] in meters and m/s
"""
function keplerian_to_cartesian(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Vector{Float64}
    af = Float64(a)
    ef = Float64(e)
    incf = Float64(inc)
    Ωf = Float64(RAAN)
    ωf = Float64(argp)
    νf = Float64(true_anom)
    μf = Float64(mu)

    # Semi-latus rectum
    p = af * (1.0 - ef^2)

    # Position and velocity in the perifocal (PQW) frame
    r_pqw = p / (1.0 + ef * cos(νf))
    x_pqw = r_pqw * cos(νf)
    y_pqw = r_pqw * sin(νf)

    sqrt_mu_p = sqrt(μf / p)
    vx_pqw = -sqrt_mu_p * sin(νf)
    vy_pqw = sqrt_mu_p * (ef + cos(νf))

    # Rotation matrix from PQW to ECI
    cosΩ = cos(Ωf); sinΩ = sin(Ωf)
    cosω = cos(ωf); sinω = sin(ωf)
    cosi = cos(incf); sini = sin(incf)

    # Column 1 of rotation matrix (P direction)
    R11 = cosΩ * cosω - sinΩ * sinω * cosi
    R21 = sinΩ * cosω + cosΩ * sinω * cosi
    R31 = sinω * sini

    # Column 2 of rotation matrix (Q direction)
    R12 = -cosΩ * sinω - sinΩ * cosω * cosi
    R22 = -sinΩ * sinω + cosΩ * cosω * cosi
    R32 = cosω * sini

    # Transform to ECI
    x  = R11 * x_pqw + R12 * y_pqw
    y  = R21 * x_pqw + R22 * y_pqw
    z  = R31 * x_pqw + R32 * y_pqw

    vx = R11 * vx_pqw + R12 * vy_pqw
    vy = R21 * vx_pqw + R22 * vy_pqw
    vz = R31 * vx_pqw + R32 * vy_pqw

    return [x, y, z, vx, vy, vz]
end

"""
    kep_x0(a, e, inc, RAAN, argp, true_anom, mu)

Return initial x-position [m] from Keplerian elements.
"""
function kep_x0(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Float64
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)
    return sv[1]
end

"""
    kep_y0(a, e, inc, RAAN, argp, true_anom, mu)

Return initial y-position [m] from Keplerian elements.
"""
function kep_y0(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Float64
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)
    return sv[2]
end

"""
    kep_z0(a, e, inc, RAAN, argp, true_anom, mu)

Return initial z-position [m] from Keplerian elements.
"""
function kep_z0(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Float64
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)
    return sv[3]
end

"""
    kep_vx0(a, e, inc, RAAN, argp, true_anom, mu)

Return initial x-velocity [m/s] from Keplerian elements.
"""
function kep_vx0(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Float64
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)
    return sv[4]
end

"""
    kep_vy0(a, e, inc, RAAN, argp, true_anom, mu)

Return initial y-velocity [m/s] from Keplerian elements.
"""
function kep_vy0(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Float64
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)
    return sv[5]
end

"""
    kep_vz0(a, e, inc, RAAN, argp, true_anom, mu)

Return initial z-velocity [m/s] from Keplerian elements.
"""
function kep_vz0(a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, true_anom::Real, mu::Real)::Float64
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, true_anom, mu)
    return sv[6]
end

"""
    constellation_ic(N, a, e, inc, RAAN, argp, mu)

Generate initial conditions for N satellites equally spaced in true anomaly
on the same orbit. Returns a Dict with keys :x0, :y0, :z0, :vx0, :vy0, :vz0,
each a Vector{Float64} of length N.
"""
function constellation_ic(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)
    x0  = zeros(N)
    y0  = zeros(N)
    z0  = zeros(N)
    vx0 = zeros(N)
    vy0 = zeros(N)
    vz0 = zeros(N)
    for i in 1:N
        nu = 2π * (i - 1) / N
        sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
        x0[i]  = sv[1]
        y0[i]  = sv[2]
        z0[i]  = sv[3]
        vx0[i] = sv[4]
        vy0[i] = sv[5]
        vz0[i] = sv[6]
    end
    return Dict(:x0 => x0, :y0 => y0, :z0 => z0,
                :vx0 => vx0, :vy0 => vy0, :vz0 => vz0)
end

# Individual IC extractors for constellation (indexed by satellite number)
# These accept literal arguments and return vectors, usable from Dyad parameter arrays
"""Return x0 array for N equally-spaced satellites on the given orbit."""
function constellation_x0(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Vector{Float64}
    return constellation_ic(N, a, e, inc, RAAN, argp, mu)[:x0]
end

function constellation_y0(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Vector{Float64}
    return constellation_ic(N, a, e, inc, RAAN, argp, mu)[:y0]
end

function constellation_z0(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Vector{Float64}
    return constellation_ic(N, a, e, inc, RAAN, argp, mu)[:z0]
end

function constellation_vx0(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Vector{Float64}
    return constellation_ic(N, a, e, inc, RAAN, argp, mu)[:vx0]
end

function constellation_vy0(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Vector{Float64}
    return constellation_ic(N, a, e, inc, RAAN, argp, mu)[:vy0]
end

function constellation_vz0(N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Vector{Float64}
    return constellation_ic(N, a, e, inc, RAAN, argp, mu)[:vz0]
end

"""Number of unique satellite pairs: N*(N-1)/2"""
function n_pairs(N::Integer)::Integer
    return div(N * (N - 1), 2)
end

"""
    pair_index(i, j, N)

Convert satellite pair (i, j) with i < j to linear index in 1:N*(N-1)/2.
Mapping: (1,2)→1, (1,3)→2, ..., (1,N)→N-1, (2,3)→N, ...
"""
function pair_index(i::Integer, j::Integer, N::Integer)::Integer
    # Row i: starts at index (i-1)*N - i*(i-1)/2 + (j-i)
    return div((i - 1) * (2N - i), 2) + (j - i)
end

# ============================================================
# Per-satellite IC extractors for use in Dyad comprehensions
# These take (sat_index, N, orbit_params...) → scalar Float64
# The sat_index and N are concrete integers from structural loops
# ============================================================

"""Return x0 for satellite `idx` (1-based) in an N-satellite constellation."""
function sat_x0(idx::Integer, N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Float64
    nu = 2π * (idx - 1) / N
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
    return sv[1]
end

function sat_y0(idx::Integer, N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Float64
    nu = 2π * (idx - 1) / N
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
    return sv[2]
end

function sat_z0(idx::Integer, N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Float64
    nu = 2π * (idx - 1) / N
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
    return sv[3]
end

function sat_vx0(idx::Integer, N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Float64
    nu = 2π * (idx - 1) / N
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
    return sv[4]
end

function sat_vy0(idx::Integer, N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Float64
    nu = 2π * (idx - 1) / N
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
    return sv[5]
end

function sat_vz0(idx::Integer, N::Integer, a::Real, e::Real, inc::Real, RAAN::Real, argp::Real, mu::Real)::Float64
    nu = 2π * (idx - 1) / N
    sv = keplerian_to_cartesian(a, e, inc, RAAN, argp, nu, mu)
    return sv[6]
end
