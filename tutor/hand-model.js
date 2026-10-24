/* hand-model.js: forward kinematics -- joint angles in, the 21 MediaPipe hand
 * landmarks out. Used to build synthetic hands the feature pipeline can be
 * tested against, and later to draw the ghost hand and the reference cards.
 *
 * FRAMES (everything downstream depends on these, so they are spelled out).
 *
 * Hand-local, where the hand is assembled: x radial (toward the thumb),
 * y distal (wrist toward the fingertips), n palmar (out of the palm, the way
 * the palm faces). Right-handed: x cross y = n. Lengths are in units of the
 * wrist -> middle-MCP distance.
 *
 * Camera, what we return: X right, Y DOWN, Z away from the camera; also
 * right-handed. The default view maps (x, y, n) -> (+X, -Y, -Z), which is a
 * right hand held palm-out with the fingers up as an UNMIRRORED camera sees
 * it: the thumb lands on image-right. Three consequences the rest of the
 * project relies on: unit(cross(P5 - P0, P17 - P0)) points out of the palm,
 * i.e. at the camera (-Z), for a right hand; flexing a finger moves its tip
 * toward -Z; and a left hand is this right hand with X negated about the view
 * center, which is what the same unmirrored camera would show.
 *
 * WHAT ROUND-TRIPS. Flexion at a joint that sits between two bones of a chain
 * -- PIP, DIP, and the thumb's MCP and IP -- is exactly the angle between those
 * two bones, so anything measuring the landmarks recovers the number that was
 * put in. A finger's MCP flexion does NOT. It is defined against the finger's
 * own distal axis (+y), while the only thing the landmarks can measure is
 * angle(P0 -> Pmcp, Pmcp -> Ppip), which takes its reference from
 * wrist -> knuckle instead. The metacarpals fan out, the resting phalanges are
 * all parallel to +y, and the difference is a per-finger rest offset: at mcp 0
 * that proxy reads index 15.87, middle 0.00, ring 14.47, pinky 29.90 degrees,
 * and at mcp 45 it reads 47.14 / 45.00 / 46.79 / 52.19. Code comparing a
 * measured MCP angle with a commanded one has to allow for that offset.
 *
 * X, Y and Z come back in one shared unit, which is what MediaPipe reports for
 * a square image. Real captures need x and z multiplied by width/height before
 * any angle is measured (see landmarks.js toPoints); a synthetic hand from
 * here is already square, so pass aspect 1.
 *
 * The proportions are generic -- rounded population-typical ratios, not any
 * participant's hand, and not measured from study data. They exist so a test
 * hand and a drawn ghost hand look like a hand; they are not a model of
 * anyone, and no result should depend on their exact values. */
import { add, cross, scale, unit } from "./vec.js";

export const FINGERS = ["index", "middle", "ring", "pinky"];

const DEG = Math.PI / 180;

// Palmar unit vector in the hand-local frame: x cross y = n = [0, 0, 1].
const PALMAR = [0, 0, 1];

// MCP (knuckle) positions in the palm plane, and phalanx lengths
// proximal -> middle -> distal. Units of wrist -> middle-MCP = 1.
const KNUCKLE = {
  index: [0.27, 0.95, 0],
  middle: [0, 1, 0],
  ring: [-0.24, 0.93, 0],
  pinky: [-0.46, 0.8, 0],
};
const PHALANX = {
  index: [0.42, 0.25, 0.2],
  middle: [0.46, 0.29, 0.21],
  ring: [0.42, 0.27, 0.21],
  pinky: [0.33, 0.19, 0.18],
};

// The thumb starts forward of the palm plane (the small n component) because a
// real thumb sits in front of it -- that offset is what lets flexion carry the
// thumb across the palm instead of through it.
const THUMB_CMC = [0.3, 0.25, 0.05];
const THUMB_BONES = [0.4, 0.33, 0.27];
const THUMB_REST = unit([0.8, 0.6, 0.25]);

// Which way positive abduction swings each finger. For the index, ring and
// pinky, positive means away from the middle finger: radial (toward the thumb)
// for the index, ulnar for the other two. The middle finger has no "away", so
// it shares the index's radial sense, and that shared sense is what lets a pose
// cross the fingers: index abd -25 with middle abd +25 sends the index tip
// ulnar-ward past the middle tip, the way a crossed-fingers handshape looks.
const ABD_SIGN = { index: 1, middle: 1, ring: -1, pinky: -1 };

const IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

export function openPose() {
  // A flat hand with the thumb spread. Every value is zero except the thumb's
  // carpometacarpal abduction, because zero there would lay the thumb along
  // the index finger, which is a handshape rather than a neutral pose.
  return {
    thumb: { cmcFlex: 0, cmcAbd: 35, mcp: 0, ip: 0 },
    index: { mcp: 0, pip: 0, dip: 0, abd: 0 },
    middle: { mcp: 0, pip: 0, dip: 0, abd: 0 },
    ring: { mcp: 0, pip: 0, dip: 0, abd: 0 },
    pinky: { mcp: 0, pip: 0, dip: 0, abd: 0 },
  };
}

// Deep merge one joint patch into a pose, copying both levels so a caller can
// keep reusing the pose it passed in.
//
// A name the pose does not have throws, at both levels. This used to merge
// anything: {index: {flex: 90}} for {index: {pip: 90}} was accepted in
// silence, the index finger came out straight, and the synthetic hand built
// from it was confidently wrong -- with nothing anywhere to say a joint had
// been ignored. The pose itself is the list of valid names, so openPose()
// decides what a finger and a joint are called and this can never drift from
// it.
export function withJoints(pose, patch) {
  if (!pose || typeof pose !== "object") {
    throw new Error(`withJoints: expected a pose object to patch, got ${pose === null ? "null" : typeof pose} -- start from openPose()`);
  }
  const out = {};
  for (const part of Object.keys(pose)) out[part] = { ...pose[part] };

  for (const part of Object.keys(patch || {})) {
    if (!Object.hasOwn(out, part)) {
      throw new Error(`withJoints: unknown finger "${part}" -- valid ones are ${Object.keys(out).join(", ")}`);
    }
    const joints = patch[part] || {};
    for (const joint of Object.keys(joints)) {
      if (!Object.hasOwn(out[part], joint)) {
        throw new Error(`withJoints: unknown joint "${joint}" on ${part} -- valid ones are ${Object.keys(out[part]).join(", ")}`);
      }
    }
    out[part] = { ...out[part], ...joints };
  }
  return out;
}

// Rotate about the palmar axis, i.e. swing within the palm plane. Positive
// degrees go toward +x (radial, toward the thumb), which is the opposite sense
// to the right-hand rule about n; abduction reads better that way, since the
// thumb side is the side people name.
function swingInPalm(v, degrees) {
  const c = Math.cos(degrees * DEG), s = Math.sin(degrees * DEG);
  return [v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]];
}

// The palmar-most direction perpendicular to `dir`: the direction a segment
// pointing along `dir` moves when it flexes. Built from the segment's own
// radial axis r = dir x n so that flexion is a rotation about r, which is the
// axis a real MCP/PIP/DIP joint turns about.
function bendDirection(dir) {
  const r = unit(cross(dir, PALMAR));
  return unit(cross(r, dir));
}

// Walk a chain of bones out from `start`. `angles[i]` is the TOTAL flexion of
// bone i measured from `dir`, so callers accumulate (mcp, mcp+pip,
// mcp+pip+dip): every bone then lies in the one plane spanned by `dir` and
// `bend`, and the angle between two successive bones is exactly the difference
// of their totals. For a joint BETWEEN two bones of the chain -- PIP, DIP, and
// the thumb's MCP and IP -- that difference is the joint value the caller asked
// for, so those round-trip exactly. The first bone's flexion (a finger's MCP,
// the thumb's cmcFlex) has no preceding bone to be measured against; see the
// header for what the landmarks can and cannot recover.
function boneChain(start, dir, bend, lengths, angles) {
  const out = [];
  let p = start;
  for (let i = 0; i < lengths.length; i++) {
    const t = angles[i] * DEG;
    p = add(p, scale(add(scale(dir, Math.cos(t)), scale(bend, Math.sin(t))), lengths[i]));
    out.push(p);
  }
  return out;
}

// A joint the caller left out rests at zero. A joint the caller set to NaN is
// a bug upstream, so let it through: NaN landmarks are obvious, a silently
// straightened finger is not.
const deg = (v) => (v === undefined || v === null ? 0 : v);

function applyMatrix(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

export function poseToLandmarks(pose, view) {
  const { hand = "right", rotation = IDENTITY, size = 0.25, center = [0.66, 0.6, 0] } = view || {};
  const local = [[0, 0, 0]]; // 0: wrist, the origin of the hand-local frame

  // Thumb (1-4: CMC, MCP, IP, tip). cmcAbd swings the whole thumb away from
  // the index within the palm plane; cmcFlex then swings it out of that plane
  // toward palmar and, because the thumb points radially, on across the palm.
  // MCP and IP flexion continue in that same plane of motion -- a real thumb's
  // three joints are near enough parallel through opposition that one plane is
  // an honest simplification, and it keeps the chain free of the gimbal case
  // where a fully opposed thumb would lose its flexion axis.
  const thumb = pose.thumb || {};
  const thumbDir = swingInPalm(THUMB_REST, deg(thumb.cmcAbd));
  const cmcFlex = deg(thumb.cmcFlex), thumbMcp = deg(thumb.mcp), thumbIp = deg(thumb.ip);
  local.push(THUMB_CMC, ...boneChain(THUMB_CMC, thumbDir, bendDirection(thumbDir), THUMB_BONES,
    [cmcFlex, cmcFlex + thumbMcp, cmcFlex + thumbMcp + thumbIp]));

  // Fingers (5-20: MCP, PIP, DIP, tip each). The finger starts along the
  // distal axis, swung in the palm plane by its abduction, and then flexes
  // toward the palm at each joint.
  for (const name of FINGERS) {
    const j = pose[name] || {};
    const dir = swingInPalm([0, 1, 0], ABD_SIGN[name] * deg(j.abd));
    const mcp = deg(j.mcp), pip = deg(j.pip), dip = deg(j.dip);
    local.push(KNUCKLE[name], ...boneChain(KNUCKLE[name], dir, bendDirection(dir), PHALANX[name],
      [mcp, mcp + pip, mcp + pip + dip]));
  }

  return local.map((p) => {
    // Hand-local -> camera axes: (x, y, n) -> (+X, -Y, -Z). That map is a
    // half turn about X, not a reflection, so cross products keep their sense
    // and the palm normal still points out of the palm.
    const v = applyMatrix(rotation, [p[0], -p[1], -p[2]]);
    const x = v[0] * size;
    // Mirroring X about the center -- and only X -- is what turns a right hand
    // into the left hand the same unmirrored camera would see. It happens after
    // `rotation`, so a left hand is the exact mirror of the right hand, tilt
    // included, and `rotation` therefore acts in mirrored camera axes for it:
    // see rotationXYZ before posing a left hand at an angle.
    return [center[0] + (hand === "left" ? -x : x), center[1] + v[1] * size, center[2] + v[2] * size];
  });
}

// Rotation applied X then Y then Z, i.e. R = Rz Ry Rx, about the CAMERA axes
// (X right, Y down, Z away) for a RIGHT hand -- camera axes rather than
// hand-local ones because the caller posing a ghost hand is looking at the
// image and wants to tip the hand within it. Two things a caller has to know.
//
// LEFT HANDS GET MIRRORED CAMERA AXES. poseToLandmarks mirrors X *after* this
// rotation, so for a left hand the effective image-frame rotation is the
// conjugate M R M^-1, with M the X mirror. A turn about X keeps its sense; a
// turn about Y or Z REVERSES it. That ordering is deliberate -- it is what
// makes a left hand the exact mirror of the right hand, tilt included -- but it
// means the same arguments tip the two hands opposite ways on screen. With
// rotationXYZ(0, 30, 0) the knuckle sitting on image-right turns toward the
// camera for a right hand (index MCP, z -0.034) and away from it for a left one
// (pinky MCP, z +0.058). A caller who wants the SAME on-screen tilt for both
// hands must negate ry and rz for the left hand: rotationXYZ(rx, -ry, -rz).
//
// PROPER ROTATIONS ONLY. poseToLandmarks multiplies by whatever 3x3 it is
// handed. A matrix that is not orthonormal with determinant +1 is accepted and
// silently distorts the hand -- diag(2, 1, 1) stretches the index proximal
// phalanx from 0.105 to 0.110 and no landmark says so -- so build the matrix
// here, or check it, rather than passing an arbitrary one through.
export function rotationXYZ(rxDeg, ryDeg, rzDeg) {
  const cx = Math.cos(rxDeg * DEG), sx = Math.sin(rxDeg * DEG);
  const cy = Math.cos(ryDeg * DEG), sy = Math.sin(ryDeg * DEG);
  const cz = Math.cos(rzDeg * DEG), sz = Math.sin(rzDeg * DEG);
  const rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]];
  const ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  return multiply(rz, multiply(ry, rx));
}

function multiply(a, b) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  }
  return out;
}
