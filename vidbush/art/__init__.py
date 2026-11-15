"""Art layer (run 3): illustrated maps, declared-style illustrations and short generated clips.

Every picture made here is openly an illustration or an animation and is counted as designed
animation (SourceKind illustrated_map, illustration, art_video). It never poses as a photograph of
a real event, place or person; photo-style AI fill stays under the stills rules (visuals/ai_images).

Modules:
- mapref: the exact map view rendered as a clean reference (land, water, borders, no text), the
  land mask, and the registration check that rejects a painting whose coastline drifted.
- paint: illustrated maps (Part B): prompt from the narration and the map's own places, variants,
  registration first, vision appeal second, plain map when none passes.
- illustrate: illustrations in the brand's one style line (Part C), judged like any still.
- clips: image to video from a chosen still, illustration or painted map (Part D), checked by a
  vision call with fixed yes or no questions, still with Ken Burns when a clip fails.
- elevate: the Elevate image call with reference images (ported from RadioPlay's request shape).
- plan: which shots get what, from the profile's art shares (Part E), and apply_art(shotlist, ctx).
- hooks: the two single-line hooks the map renderer calls (painted base, moving base).
"""
