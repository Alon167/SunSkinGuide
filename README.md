# Muscular teddy-bear boxer – 3D-printable (P1S, 0.4 mm nozzle, PLA)

**Built:** a 200 mm plush-style boxer bear (SDF-modelled in Python, revision 2: rounded/organic masses, relaxed forward boxer pose): plush-tuft-textured brown body/legs, smooth glossy red gloves
and black shorts, separate cream muzzle / black nose / black eyes with white highlight pins, optional Ø96 base.
All structural joints are hidden short split-snap pins (printed in the part itself). Fur = real 0.35–0.6 mm bumps (FUR_STRENGTH).
Face is deliberately softer than the reference: brows ~10° slant, eyes fully visible, almost straight mouth.

Regenerate everything: `pip install numpy scipy scikit-image trimesh manifold3d numba matplotlib pillow rtree && python generate.py`
(~5 min). Options: `--fur 0` (smooth), `--fur 0.6`, `--clearances 0.12,0.18,0.24`, `--height 180` (uniform scale of everything incl. pins),
`--no-base`, `--preview` (fast), `--set TORSO_PIN=12,9 --set BARB=0.35`. Parameters live in `bear/config.py`.

## Parts (STL files are already in print orientation)
| # | file | colour | qty | orientation | supports |
|---|------|--------|-----|-------------|----------|
| 01 | clearance_X/01_body_brown | brown | 1 | reclined on its back (rotated 65° about X from upright, STL already posed): face/chest face up, torso pin 25° and wrist pins ~29° off horizontal | yes, tree supports on the back and under the upper arms/forearms (≈8,300 mm² flagged, mostly hidden back); face, chest, abs support-free |
| 02 | common/02_muzzle_cream | cream | 1 | flat back down | no |
| 03 | common/03_nose_black | black | 1 | flat back down | no |
| 04 | common/04_eye_black | black | 2 (+2 spare) | flat plug down, dome up | no |
| 05 | common/05_highlight_white | white | 2 (+4 spare) | standing | no (use brim) |
| 06/07 | 06_glove_left_red / 07_glove_right_red | red | 1 each | cuff face down | small support under thumb tip optional |
| 08 | 08_shorts_black | black | 1 | hem down | none needed (fold/crotch overhangs are shallow) |
| 09/10 | 09_leg_left_brown / 10_leg_right_brown | brown | 1 each | on the flat sole | no |
| 11 | 11_base_black | black | 1 (optional) | flat | no |

Layer height: 0.12 mm for textured parts (body, legs), 0.12 gloves/shorts/base, 0.08 mm for muzzle, nose, eyes, highlights.
`common/with_brim/` has the tiny parts with a 0.24 mm × 1.6 mm peelable brim (already used in the 3MFs).
Each `clearance_X/plate_clearance_X.3mf` holds every part incl. spares on one 256×256 plate with colour names (assign filaments per colour).
Left/right = figure's own left/right.

## Choosing the clearance
Print `test_snap.stl` (≈10–15 min): top block has sockets for 0.10/0.15/0.20/0.25 mm per side, the pin plate has 4 matching vertical pins ("V"),
the small block "H" is a pin printed lying down. Pick the value that clicks firmly but needs no tools, then print everything from the
folder `clearance_0.10`, `clearance_0.15` or `clearance_0.20` (nearest value; regenerate with `--clearances` for others).

## Assembly (permanent click joints, no glue except face parts)
1. Legs 09/10 → push their top pins up into the shorts' hem sockets (click, 4 mm collar hides the seam).
2. Body 01 (torso pin Ø12×9, flat to the back) → press down into the shorts' waistband.
3. Gloves 06/07 → wrist pin Ø8×6 + 4 mm collar into the glove cuff (click).
4. Base 11: its two pins click into the foot soles (or skip the base – soles are flat).
5. Glue (one drop): eyes (Ø9.6 plug, recess 1.5 mm), muzzle (1 mm recess, locating post/hole), nose into the muzzle (0.8 mm recess, triangular key),
   highlight pins into the Ø2.0 holes (upper-left of each eye – same side on both).

## Notes
* `assembled_preview.stl` = whole figure for viewing only. `build_report.json` has the automatic checks (see DECISIONS.md).
* Pose: upper arms hang ~16° outward, elbows bent ~36°, forearms/gloves point forward-down; gloves sit ~17 mm in front of the chest and ~15 mm from the shorts.
* Fur: plush tufts (3–4 merged bumps, 0.5–0.7 mm high, ≥0.86 mm wide) flowing downward (face: outward). `--fur 0` = smooth.
* Version 1 renders are kept in `output_v1_renders/`; comparison sheets against reference.png are `output/renders/compare_*.png`.
* Renders: `output/renders/` (4 views + 45° views, face, fur, exploded).
