# ZPK release contract

`package.meta` describes the package identity and the service that OTA should
operate on after installation. For `zettlab-claw`, `service_name` must continue
to match the installed systemd unit name without the `.service` suffix:
`zettlab-claw`.

Restart behavior is controlled by the OTA release/package record, not by a
`restart` key in `package.meta`. When a published `zettlab-claw` package must
take effect immediately after upgrade, publish that package record with
`restart=1`; OTA will restart the service named by `package.meta.service_name`
after installing the package.

Do not add package-local restart metadata unless `zettlab-ota` intentionally
adopts `package.meta` as a supported restart source of truth.
