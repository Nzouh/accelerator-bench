# HTP preparation audit

Preparation flags are now explicit on inference jobs. The original sweep supplied them only at DLC conversion; effective HTP values were unverified.

Case: `conv_b1_c32_h16_o64`. Same inputs, device and QAIRT. One job/request; no latency claim.

| Request | Legacy graph configs | Explicit graph configs | HTP nodes, legacy → explicit | Same HTP op counts |
|---|---:|---:|---:|---|
| O3 | 3 | 3 | 47 → 47 | Yes |
| O1 | 3 | 3 | 47 → 24 | No |
| O3 / no fold | 3 | 4 | 47 → 47 | Yes |

Configuration counts show setup differences, not individual option values. HTP op counts do not establish binary identity.

[Evidence and log excerpts](../../results/hub-s24-preparation-audit-2026-10-14/preparation-review.json) · [Preparation stages](https://workbench.aihub.qualcomm.com/docs/hub/api.html)
