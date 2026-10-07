# DECISIONS

## NEEDS USER
* Nothing was skipped, but **nothing was test-printed or sliced** (no printer/slicer here). Print `test_snap.stl` first.
* Open risk: barb 0.40 mm on 8–12 mm pins with 70 % slots is aggressive for PLA (estimated bending strain well above PLA's ~2–3 %
  yield on paper). It may need `--set BARB=0.30`. Leg pins print vertical (as requested: legs stand on their soles) and are the weakest joint;
  if one snaps, print that leg on its side with tree supports.
* Bambu Studio may ignore 3MF base colours; assign filaments by object name/colour.

## Decisions (reason)
* SDF + numba, fur via Poisson-disk seeds → sunk-sphere bumps (even coverage, per-area strength, ellipsoid flow elongation) – Worley noise gave uneven sizes on surfaces.
* Fur bump spacing 0.70×(Ri+Rj) → ~0.87 bumps/mm²; fBm 0.15 mm on top; grooves halve amplitude; chest/abs 70 %, inner ear 50 % finer, soles/joints 0 (2 mm smooth zone + ramp).
* Muzzle velvet is built but is ≈0.1 mm high, i.e. barely visible in PLA (physically below nozzle resolution).
* Voxel 0.25 mm (fur), 0.18–0.22 (smooth), 0.12 (muzzle); mesh simplified with manifold3d tolerance 0.01–0.035 mm (keeps manifold; body ≈77 MB < 80 MB).
* Cut faces: vertices near/over mating planes are snapped onto the plane after meshing (marching cubes rounds edges) → measured assembled interference ≤2.6 mm³ total.
* Joints: pin on body/wrist/leg-top/base, sockets in shorts/glove/sole; barb 0.4, lead-in 35°, flat retaining face, slot 1.3 × 70 %, fillet 1.0 (1.5 base),
  entrance chamfer 0.5, socket +0.5 deeper, anti-rotation flat 1 mm; extra relief ring so the fillet clears the socket; groove ceiling 45° (printable). Wrist and leg joints get a
  4 mm collar (tongue) so seams are hidden. Torso seam sits on the waistband top. Measured pin-vs-socket interference = 0.000 mm³ for clearances 0.10/0.15/0.20.
* Body is printed lying on its back (pins horizontal = layers along the prong); chest/face support-free; back needs supports (~7200 mm² flagged, hidden side). Best tilt searched ±10°.
* Muzzle locating feature: post on the head recess + hole in the muzzle back (a post on the muzzle would prevent printing it flat). Nose is keyed by its triangular shape.
* Muzzle/eye recess floors are flat planes (min depth 1.0 / 1.5 mm at the shallowest point; deeper at centre) so the parts print flat.
* Highlight pin length = hole depth 1.4 mm (flush). 4 eyes (2 spare), 6 highlights (2 + 4 spare).
* Base Ø96 (min circle around both soles + 4 mm; "about 90"), centred on the feet.
* `--height` is a plain uniform scale (pins/clearances scale too); only 200 mm was verified.
* Printable-wall (≥1.2 mm) was checked by design (socket walls ≥ 4 mm) not by a full wall-thickness scan.

## Improvement rounds (renders vs reference.png)
1. First coarse render: head boxy, ears small, pecs/abs brick-like, shorts folds looked like cracks, back "arrow" grooves, legs thin → rounder head, larger higher ears, ellipsoid pecs, softer abs, thicker arms/legs/deltoids, tapered fabric folds, subtle scapula/lat grooves.
2. Face: brows made more visible (still ≥2 mm clear of the eyes), eye dish smaller so fur stays near the eyes; compared with reference the face is clearly less angry (round eyes, ~10° brows, flat mouth).
3. Shorts: removed diagonal crotch folds (looked like gouges), widened hems; fur density raised (0.78→0.70 spacing factor); fixed overlaps at seams (plane snapping), relief ring on sockets, hem torus no longer pokes below the hem plane.
Remaining differences vs reference: shorts are plainer/less shiny-wrinkled, fur reads finer, arms hang a bit more forward, ears a little smaller.
