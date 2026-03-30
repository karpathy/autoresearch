# Heavy Construction & Industrial Equipment Research

## Objective

Compile a categorized inventory of every type of equipment, vehicle, machine, and tool used on a large-scale commercial construction site. For each item, provide the common name, aliases, a brief description, typical use phase, and a direct image URL from a manufacturer or industry source.

## Scope

Cover all phases of construction from site preparation through finishing, plus supporting vehicles, specialty trucks, trade-specific tools, agricultural/forestry crossover equipment, and emergency/utility vehicles present on active job sites.

## Instructions

Research **Phase ${phase}** below. List every distinct equipment type within each category in that phase.

**Depth: ${depth}**

- `seed` -- list only items explicitly named in the category
- `expanded` -- add all sub-types and variants beyond the seed list
- `exhaustive` -- expanded + cross-reference manufacturer catalogs (Caterpillar, Komatsu, Volvo CE, John Deere, Liebherr, SANY, Hilti, DeWalt, Milwaukee, Ridgid, Greenlee, Klein Tools, Lincoln Electric, Miller Electric) and rental inventories (United Rentals, Sunbelt, Herc)

## Rules (apply to every phase)

1. Do not skip items because they seem too common or too niche.
2. If equipment has major sub-types that look or function differently, list each sub-type as a separate row.
3. For image URLs, prefer manufacturer product pages, Wikimedia Commons, or major rental company listings. Use the most recognizable version of each machine.
4. After completing the phase, add an "Unlisted / Miscellaneous" section for any equipment found during research that fits this phase but not the listed categories.

## Output Format

Group rows under their category heading. Each row follows this structure:

```text
### {Category Name}

| # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |
|---|----------------|---------------|-------------|-------------------|-----------|
| 1 | {name}         | {aliases}     | {1-2 sentence function description} | {site prep / foundation / structural / MEP / finishing / all phases} | {direct image URL} |
```

---

## Phase 1: Heavy Equipment

> Output each item as a table row: | # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |

### 1. Earthmoving & Excavation

Excavators (standard, mini/compact, long-reach, spider, amphibious, dragline, suction), bulldozers (crawler, wheel, mini), backhoe loaders, front-end loaders (wheel, track, skid steer, compact track), scrapers (pull, self-propelled), motor graders, trenchers (chain, wheel, micro), compactors/rollers (vibratory, pneumatic, sheepsfoot, smooth drum, plate compactor, rammer/jumping jack), dump trucks (standard, articulated, rigid, side dump), dumpers (site dumpers, track dumpers).

### 2. Lifting & Material Handling

Cranes (tower, mobile/truck, crawler, rough terrain, all-terrain, overhead/bridge, gantry, mini spider, knuckle boom, telescopic), forklifts (warehouse, rough terrain, telehandler/telescopic handler), aerial work platforms (scissor lifts, boom lifts -- articulating and telescopic, personnel lifts, mast climbers, suspended scaffolding platforms), hoists (material, personnel), conveyors (belt, screw, bucket elevator).

### 3. Concrete & Masonry

Concrete mixer trucks (standard drum, volumetric), concrete pumps (boom pump, line pump, separate placing boom), concrete vibrators (internal/poker, external, surface/screed), concrete saws (walk-behind, handheld), power trowels (ride-on, walk-behind), concrete batching/mixing plants (mobile, stationary), shotcrete machines, grout pumps, mortar mixers, block saws.

### 4. Paving & Road Construction

Asphalt pavers (track, wheel), asphalt rollers, cold planers/milling machines, asphalt distributors, chip spreaders, slipform pavers, road reclaimers, line striping machines, crack sealing machines.

### 5. Pile Driving & Foundation

Pile drivers (diesel hammer, hydraulic hammer, vibratory, drop hammer), pile boring/drilling rigs (rotary, CFA, kelly bar), sheet pile drivers, diaphragm wall equipment (hydromills, clamshell grabs), jet grouting rigs, micropile drills.

### 6. Demolition

Wrecking balls, hydraulic breakers/hammers (excavator-mounted, handheld jackhammers), concrete crushers/pulverizers (excavator-mounted), demolition robots, demolition shears, wall saws, wire saws, diamond chain saws.

### 7. Drilling & Boring

Rock drills (top hammer, down-the-hole), horizontal directional drills (HDD), auger boring machines, tunnel boring machines (TBM), raise boring machines, core drills, rotary blasthole drills, rock bolting rigs.

---

## Phase 2: Vehicles & Transport

> Output each item as a table row: | # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |

### 8. Commercial & Specialty Trucks

Dump trucks (on-highway), concrete mixer trucks, flatbed trucks, lowboy/lowbed trailers, semi-trucks/big rigs (day cab, sleeper), water trucks, fuel/tanker trucks, vacuum trucks (hydrovac), boom trucks/crane trucks, knuckle boom loader trucks.

### 9. Utility & Service Vehicles

Bucket trucks/aerial trucks, cable laying trucks, digger derrick trucks, service/mechanic trucks, welding trucks, light towers (trailer-mounted), mobile generator trailers, mobile compressor trailers.

### 10. Emergency & On-Site Safety Vehicles

Fire engines/pumper trucks, aerial ladder trucks, rescue trucks, ambulances/paramedic units, hazmat response vehicles, mobile command centers.

### 11. Waste & Environmental

Garbage trucks (rear loader, front loader, side loader, roll-off), street sweepers (mechanical, vacuum, regenerative air), dust suppression units, silt fence installers, hydroseeding trucks.

### 12. Transport & Logistics

Car carriers/auto transport trailers, equipment transport lowboys, container chassis trailers, delivery vans/box trucks, cargo vans, pickup trucks (1/2-ton through 1-ton), flatbed delivery trucks, airport tugs/pushback tractors, aircraft tow tractors, baggage tractors, ground power units.

---

## Phase 3: Specialty & Land Clearing

> Output each item as a table row: | # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |

### 13. Forestry & Land Clearing

Forest harvesters (wheeled, tracked), forwarders, feller bunchers, log skidders, wood chippers/grinders (industrial), stump grinders, mulchers (excavator-mounted, skid steer-mounted), brush cutters.

### 14. Agricultural Crossover

Tractors (utility, compact, row crop), combine harvesters, front-end loader tractors.

### 15. Scaffolding & Temporary Structures

Scaffolding systems (tube and clamp, frame, ringlock, kwikstage), shoring systems (vertical, horizontal), formwork systems (wall, slab, column), temporary fencing, portable offices/job trailers.

---

## Phase 4: Trade Tools

> Output each item as a table row: | # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |

### 16. Plumbing & Pipefitting Tools

Pipe threaders (manual, electric), pipe cutters (chain, rotary, reciprocating saw), pipe wrenches (Stillson, chain, strap, basin, offset), pipe benders (manual, hydraulic), pipe vises, soldering/brazing torches, press fitting tools (Viega, ProPress), drain cameras/inspection systems, hydrostatic test pumps, pipe fusion machines (butt, electro, socket), pipe lasers, conduit benders.

### 17. Electrical Trade Tools

Wire pullers/cable tuggers, cable cutters (ratchet, hydraulic), wire strippers (manual, automatic), crimping tools (standard, hydraulic), conduit benders (hand, electric, hydraulic), fish tapes/rods, multimeters, insulation resistance testers (meggars), thermal imaging cameras, cable locators/tracers, knockout punch sets, cable tray cutters.

### 18. Ironworking & Structural Steel Tools

Rebar tying tools (automatic, manual), rebar cutters (manual, electric, hydraulic), rebar benders (manual, electric, hydraulic), structural bolt wrenches (spud wrenches, sleever bars), impact wrenches (pneumatic, electric, hydraulic torque), shear connectors/stud welders, magnetic drill presses, beam clamps, come-alongs/chain hoists, rigging hardware (shackles, turnbuckles, wire rope clips, swivels, spreader beams).

### 19. HVAC Tools

Refrigerant recovery machines, vacuum pumps, manifold gauge sets, tube benders (manual, electric), flaring tools, swaging tools, sheet metal brakes (hand, hydraulic), sheet metal shears (manual, electric, pneumatic), duct forming machines, crimping tools, HVAC gauges, leak detectors, anemometers.

### 20. Welding & Cutting

Arc welders (stick/SMAW, MIG/GMAW, TIG/GTAW, flux-cored/FCAW), plasma cutters, oxy-fuel cutting torches, welding positioners/manipulators, pipe beveling machines, spot welders, stud welders, welding fume extractors.

---

## Phase 5: General Tools, Survey & Site Support

> Output each item as a table row: | # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |

### 21. General Power Tools (Construction-Grade)

Rotary hammers/hammer drills, demolition hammers (electric), angle grinders (4.5", 7", 9"), circular saws (worm drive, sidewinder), reciprocating saws, miter saws (sliding compound), table saws (jobsite), band saws (portable), concrete cut-off saws (gas, electric), powder-actuated tools (Hilti, Ramset), impact drivers, impact wrenches, magnetic drill presses, core drills (handheld), rebar cutters (handheld), channel/strut cutters.

### 22. Surveying & Layout

Total stations, GPS/GNSS receivers (rover, base), robotic total stations, laser levels (rotary, line, dot), automatic/optical levels, theodolites, measuring wheels, laser distance meters, pipe lasers, grade rods, survey prisms.

### 23. Safety & Personal Protective Equipment (Major Items)

Fall protection systems (harnesses, lanyards, SRLs, horizontal lifelines), confined space entry equipment (tripods, winches, gas monitors), respiratory protection (half-face, full-face, PAPR, SCBA), arc flash PPE kits, hearing protection (electronic muffs, custom-molded plugs).

### 24. Site Support & Miscellaneous

Portable toilets, dewatering pumps (submersible, trash, wellpoint), generators (portable, towable), air compressors (portable, towable), pressure washers (gas, electric), heaters (forced air, radiant, ground thaw), portable lighting/light towers, traffic control devices (arrow boards, message signs, barriers).
