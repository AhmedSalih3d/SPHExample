#let report = (
  simulation_name: "{{SIMULATION_NAME}}",
  completion: "{{COMPLETION}}",
  final_time: "{{FINAL_TIME}}",
  steps: "{{STEPS}}",
  total_particles: "{{TOTAL_PARTICLES}}",
  fluid_particles: "{{FLUID_PARTICLES}}",
  wall_time: "{{WALL_TIME}}",
  timing_note: "{{TIMING_NOTE}}",
)

#let navy = rgb("#17324d")
#let teal = rgb("#0f766e")
#let pale_teal = rgb("#e8f4f2")
#let pale_blue = rgb("#eef3f8")
#let border = rgb("#cfdae5")
#let muted = rgb("#586b7d")
#let good = rgb("#15803d")
#let warning = rgb("#b45309")
#let status_fill = if report.completion == "Completed" { good } else { warning }

#set page(paper: "a4", margin: (x: 16mm, y: 15mm))
#set text(size: 9pt, fill: navy)
#set par(leading: 0.72em)
#set heading(numbering: none)
#set table(
  inset: (x: 6pt, y: 4pt),
  stroke: 0.45pt + border,
  fill: (x, y) => if y == 0 { navy } else { none },
)
#show heading.where(level: 1): set text(size: 17pt, fill: navy)
#show heading.where(level: 2): set text(size: 11pt, fill: teal)
#show table.header: set text(fill: white, weight: "bold")

#align(center)[
  #text(size: 9pt, weight: "bold", fill: teal)[SPHEXAMPLEGPU / SIMULATION REPORT]
  #v(5pt)
  #text(size: 24pt, weight: "bold", fill: navy)[#report.simulation_name]
  #v(3pt)
  #text(size: 10pt, fill: muted)[GPU simulation run]
]

#v(12pt)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  block(fill: status_fill, inset: 9pt, radius: 4pt)[
    #text(size: 7pt, weight: "bold", fill: white)[STATUS]
    #linebreak()
    #text(size: 12pt, weight: "bold", fill: white)[#report.completion]
  ],
  block(fill: pale_blue, inset: 9pt, radius: 4pt)[
    #text(size: 7pt, weight: "bold", fill: muted)[FINAL TIME]
    #linebreak()
    #text(size: 12pt, weight: "bold")[#report.final_time s]
  ],
  block(fill: pale_teal, inset: 9pt, radius: 4pt)[
    #text(size: 7pt, weight: "bold", fill: muted)[STEPS]
    #linebreak()
    #text(size: 12pt, weight: "bold")[#report.steps]
  ],
  block(fill: pale_blue, inset: 9pt, radius: 4pt)[
    #text(size: 7pt, weight: "bold", fill: muted)[FLUID PARTICLES]
    #linebreak()
    #text(size: 12pt, weight: "bold")[#report.fluid_particles]
  ],
)

#v(7pt)
#text(size: 8pt, fill: muted)[Wall time: #report.wall_time s | Total particles: #report.total_particles. This report organizes the run configuration, progress, memory and timing data from the same simulation into readable sections.]

= Executive summary

#table(
  columns: (1.15fr, 1.4fr),
  table.header([Run measure], [Result], repeat: true),
  {{SUMMARY_ROWS}}
)

= Final fluid results

#table(
  columns: (1.15fr, 0.9fr, 0.9fr, 0.9fr),
  table.header([Variable], [Minimum], [Mean], [Maximum], repeat: true),
  {{VARIABLE_ROWS}}
)

#block(fill: pale_teal, inset: 8pt, radius: 4pt)[
  #text(size: 8pt)[Density describes fluid mass per volume. Pressure is calculated from final fluid density with `EquationOfStateGamma7`; speed is the magnitude of the velocity vector. Boundary particles are excluded from all three summaries.]
]

= Component guide

#table(
  columns: (1.1fr, 2.4fr),
  table.header([Component], [Description], repeat: true),
  [Physical constants], [Set the scales used by the SPH equations: particle spacing and mass, reference density, gravity, sound speed, viscosity and equation-of-state parameters.],
  [Simulation metadata], [Controls requested duration, output cadence and variables, file export, logging, numerical modes and GPU execution options.],
  [SPH kernel], [Weights neighboring particles by distance to estimate density, pressure and other interaction terms.],
  [Viscosity model], [Adds momentum diffusion between particles; the selected model and its parameters are listed in the model tables.],
  [Density diffusion], [Applies the selected density correction to help control density errors and pressure noise.],
  [Geometry and particle groups], [Identify each generated or file-backed group by marker and particle type. Fluid particles move; boundary groups define the walls or prescribed motion.],
  [Requested GPU options], [Record what the run asked the GPU driver to use.],
  [Supported specializations], [Show which optimized interaction paths are implemented for the selected models.],
  [Effective configuration], [Show the numerical operators and launch settings actually selected by the driver.],
  [Fallbacks and notices], [Describe unsupported requests, configuration fallbacks and warnings recorded during the run.],
  [Memory accounting], [Separate host particle payload, live GPU array payload, CUDA pool usage and output staging capacity; these are not interchangeable VRAM totals.],
  [Progress history], [Each row records output frame and physical time, total and interval steps, elapsed wall time, runtime per simulated second and estimated finish time.],
  [Phase timings], [Report measured solver sections. Unsynchronized GPU timings reflect host enqueue/readback boundaries rather than individual kernel durations.],
)

= Environment and hardware

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Environment item], [Value], repeat: true),
  {{ENVIRONMENT_ROWS}}
)

= Physical model and run settings

== Physical constants

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Parameter], [Value], repeat: true),
  {{CONSTANTS_ROWS}}
)

== Simulation metadata

#text(size: 8pt, fill: muted)[Runtime counters are summarized above and in progress history. Configuration fields below describe the requested case and GPU options.]

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Setting], [Value], repeat: true),
  {{METADATA_ROWS}}
)

= Particle layout

== Geometry groups

#table(
  columns: (0.55fr, 0.65fr, 0.55fr, 1.6fr, 1.5fr),
  table.header([Marker], [Type], [Count], [Source], [Motion / body], repeat: true),
  {{GEOMETRY_ROWS}}
)

== Particle type totals

#table(
  columns: (0.8fr, 0.55fr, 2.1fr),
  table.header([Type], [Count], [Role], repeat: true),
  {{PARTICLE_TYPE_ROWS}}
)

= SPH models

== Kernel

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Property], [Value], repeat: true),
  {{KERNEL_ROWS}}
)

== Viscosity

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Property], [Value], repeat: true),
  {{VISCOSITY_ROWS}}
)

== Density diffusion

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Property], [Value], repeat: true),
  {{DIFFUSION_ROWS}}
)

= Effective GPU configuration

== Requested options

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Option], [Requested value], repeat: true),
  {{REQUESTED_ROWS}}
)

== Supported specializations

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Capability], [Support], repeat: true),
  {{CAPABILITY_ROWS}}
)

== Effective settings

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Setting], [Selected value], repeat: true),
  {{EFFECTIVE_ROWS}}
)

== Fallbacks and observed execution

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Fallback], [Detail], repeat: true),
  {{FALLBACK_ROWS}}
)

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Execution measure], [Observed value], repeat: true),
  {{EXECUTION_ROWS}}
)

= Solver diagnostics

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Diagnostic], [Value], repeat: true),
  {{DIAGNOSTIC_ROWS}}
)

= Memory accounting

#table(
  columns: (1.1fr, 1.25fr, 1.6fr),
  table.header([Phase], [Buffer / measure], [Recorded value and scope], repeat: true),
  {{MEMORY_ROWS}}
)

#block(fill: pale_blue, inset: 8pt, radius: 4pt)[
  #text(size: 8pt)[Host array payload, live GPU array payload, CUDA allocator pool and output staging describe different memory scopes. Shared GPU arrays are counted once. The logger's scope note identifies excluded context, graph, allocator-reservation and allocation-overhead costs; do not read these rows as process or peak VRAM.]
]

= Performance and phase timings

== Run performance

#table(
  columns: (1.15fr, 1.85fr),
  table.header([Measure], [Value], repeat: true),
  {{PERFORMANCE_ROWS}}
)

== Timer breakdown

#table(
  columns: (1.7fr, 0.55fr, 0.75fr, 0.8fr),
  table.header([Phase / subphase], [Calls], [Time (s)], [Allocated], repeat: true),
  {{TIMING_ROWS}}
)

#text(size: 8pt, fill: muted)[#report.timing_note]

= Warnings and notices

#table(
  columns: (0.8fr, 2.2fr),
  table.header([Kind], [Recorded message], repeat: true),
  {{NOTICE_ROWS}}
)

#pagebreak()

= Output progress history

#text(size: 8pt, fill: muted)[This table reformats the progress columns written for each output frame. The runtime ratio is elapsed wall-clock seconds per simulated physical second; it is not the timestep.]

#table(
  columns: (1.2fr, 1.1fr, 1.5fr, 1.3fr),
  table.header([Frame / simulation time], [Steps (total / interval)], [Elapsed / runtime ratio], [Estimated finish], repeat: true),
  {{PROGRESS_ROWS}}
)
