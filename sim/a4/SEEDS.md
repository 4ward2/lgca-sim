# A4 seed ledger — candidate protocol, 2026-10-30

Registered before first use. NOTES.md scanned: none of these new values occurs
as a registered seed. All newly allocated seeds must be at least 100000:
the brief's `use >=100000` conflicts with the explicit spec seeds below;
therefore all newly allocated model/pilot/bootstrap values are shifted +100000.
The protocol file will carry this mapping before any random operation.

| Namespace | Spec | Allocated | First use |
|---|---|---|---|
| formal models, all six arms | 73101–73110 | 173101–173110 | remote only |
| pilot models | 73001–73003 | 173001–173003 | remote only |
| bootstrap | 73901 | 173901 | fixture statistics / remote |
| fallback train/dev generation | 731032 | 731032 | remote fallback only |
| fallback OOD generation | 732000+L | unchanged | NOT USED in this task |
| density tables | 740000+100*i+j, i,j=0..9 | unchanged | fixtures / remote |
| upstream split | 42 | 42 (inherited upstream, not introduced) | split compatibility test |

Tests use explicit bit enumeration and registered density/model generators;
no implicit default RNG seed is permitted. Forbidden: 12345 and 24680.
