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


# REVISION 2 (shape + arm pose) — decisions and checklist
Version-1 renders were copied to `output_v1_renders/` first.
* All boxy primitives removed (rounded boxes → ellipsoids; abs are pillow ellipsoids; glove/foot/head from overlapping ellipsoids). Blends: 8–10 mm for head/torso masses, 3–6 mm muscles, 1–2.5 mm details. Only flat faces left are the joint/sole faces (verified by a coplanar-facet scan: only cut faces > 400 mm²).
* Head: fuller, cheeks low/wide, jaw ellipsoid + under-chin fill (no ledge), trapezius slope capsules blend head into shoulders.
* Torso V: shoulders/waist width ratio 1.5 (95.5 mm vs 63.5 mm on the arm-less torso), chest 82.5 mm.
* Arms: shoulder rolled forward (y -2), upper arm 22 mm hanging 16° outward, forearm 27 mm bent ~36° forward-down; glove axis = forearm axis; wrist pin/collar/socket are coaxial (joint uses the same axis). Delt radius ~18, bigger biceps/triceps/forearms.
* Gloves: no finger lines; one thumb on the upper inner side; thick cuff band with seam ring + stitched pit row (Ø1.1 mm pits), shallow wrinkles near thumb/knuckles. Slightly bigger (≈44 × 37 mm) than first build to read like the reference.
* Shorts: wider hips, flared legs (r 17→24 mm), waistband with 3 rounded ribs, rolled hems, inverted-V crotch arch, hem notch; dents/streaks removed (gathers are now 9 very broad shallow waves). Mesh check: winding consistent, positive volume, 0 degenerate faces on shorts.
* Legs: thicker (r 14.8→19.6), stance 22 mm (was 20), paws with 4 toe bumps + 3 grooves.
* Fur: tufts via Poisson seeds (spacing ~1.6–2.0 mm) each expanded to 3–4 merged elongated bumps along the flow direction; height 0.5–0.7 mm; fBm 0.15 mm kept; all area rules unchanged.
* Print orientation search now balances overhang area against pin tilt: body reclined 65° (not 90°) → pin axes ≤29° from horizontal; overhang 8,266 mm² (back/arms). Forearm undersides were NOT verified in a slicer.
* Glove/chest spacing vs the brief: glove centre is ~17 mm in front of the chest (brief 20–30) and 15 mm from the shorts (brief 5–10); both together are geometrically incompatible with a 200 mm chibi arm length unless the arms were lengthened further, so I prioritised the look and measured gaps.
* Shoulder width is now ~128 mm (brief ~120) because "massive deltoids" was requested.

## Rounds and checklist
Round 1 (first rebuild): arms forward but gloves too high (z 87) and too far from shorts (gap ≈20 mm); shorts had dent-like dimples; gloves read as balls. Checklist: boxy? no. V? yes (1.5). Gloves in front at 90°? yes. Smooth/no finger lines? yes. Shorts free of dents? NO (dimples). Legs short/chunky? yes.
Round 2: lowered/lengthened arm chain (glove centre z 77), removed broad dents; added ridges under waistband → looked like pegs/claws (rejected). Checklist: boxy no; V yes; gloves in front yes; smooth gloves yes; shorts streaks: pegs (NO); legs yes.
Round 3 (final): ridges replaced by broad shallow waves, hips/waistband widened, gloves enlarged, arms moved ~3 mm inward. Checklist final: boxy/flat → no (only joint faces); V shape → yes (ratio 1.5); gloves in front in 90° view → yes (≈17 mm in front of chest, tips ≈73 mm forward of origin); gloves smooth, one thumb, no finger lines → yes; shorts free of streaks/dents → yes in renders, mesh defects: none (winding ok, volume >0); legs short and chunky → yes.
Remaining differences vs reference: fur reads finer than the reference's loops, muscles slightly less bulky, shorts less glossy-wrinkled, gloves 5–10 % smaller, glove is not stitched on every panel.
