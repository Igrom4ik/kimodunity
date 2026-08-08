# ARDY Bridge Protocol v1

Transport: TCP on `127.0.0.1:8801`, one active client, `TCP_NODELAY` enabled. Every integer and float is little-endian.

## Frame

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | `uint32` | magic `0x59445241`, bytes `ARDY` |
| 4 | 2 | `uint16` | protocol version, currently `1` |
| 6 | 2 | `uint16` | message type |
| 8 | 4 | `uint32` | UTF-8 JSON object length |
| 12 | 4 | `uint32` | binary blob length |
| 16 | N | bytes | compact UTF-8 JSON object |
| 16+N | M | bytes | optional binary blob |

Maximum accepted sizes in v1: 1 MiB JSON, 64 MiB blob. A receiver must read exactly the declared lengths; TCP packet boundaries have no protocol meaning.

## Python to Unity

| Type | Name | JSON | Blob |
|---:|---|---|---|
| 1 | `hello` | `protocol`, `model`, `skeleton`, `fps`, `jointNames`, `parents`, `rootIdx`, `footContactJoints`, `genHorizon`, `numFramesPerToken` | `neutral_joints[J,3]` as `float32` |
| 2 | `chunk` | `startFrame`, `count`, `revision`, `hasContacts` | per frame: `root[3]`, `quat[J,4]`, optional `contacts[4]`, all `float32` |
| 3 | `playhead` | `frame`, `playing`, `maxFrame` | none |
| 4 | `invalidate` | `fromFrame` | none |
| 5 | `status` | `state`, `prompt`, `lastStepMs`, `vramMb`, `message` | none |
| 6 | `error` | `code`, `message` | none |

Quaternion order is `(x, y, z, w)` from `scipy.spatial.transform.Rotation.as_quat()`. Values remain in ARDY RH/Y-up/+Z-forward coordinates; Unity performs the RH-to-LH conversion. Contacts use four `float32` values in `[L_heel, L_toe, R_heel, R_toe]` order.

For an autoregressive result containing a history prefix, Python removes the history frames before creating `chunk`; `startFrame` is the absolute index of the first newly generated frame.

`revision` increments on restart. `invalidate.fromFrame` removes that absolute frame and everything after it before replacement chunks arrive.

## Unity to Python

| Type | Name | JSON |
|---:|---|---|
| 100 | `setPrompt` | `text`, `fromFrame` |
| 101 | `transport` | `action`, optional `frame` |
| 102 | `setParams` | `diffusionSteps`, `cfgText`, `cfgConstraint`, `numSamples`, `postprocess` |
| 103 | `waypoint` | `frame`, `x`, `z` in Unity coordinates |
| 104 | `requestRange` | `fromFrame`, `toFrame` |
| 105 | `bye` | empty JSON object |

Unknown message types may be ignored after their declared payload is consumed. An unsupported protocol version, invalid length, invalid JSON, or malformed typed payload is a protocol error.

## Smoke test without the model

From the repository root, start `python tools/bridge_smoke_server.py`, then run `python tools/bridge_smoke_client.py` in another terminal. The server sends a cskel27 `hello` and a 40-frame `chunk` built from `tools/golden_transform.json`, then exits after the client sends `bye`.
