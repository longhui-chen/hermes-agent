import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'

import { ListRow } from './primitives'

export type DeepMemoryMode = 'off' | 'smart' | 'always'

export function normalizeDeepMemoryMode(value: unknown): DeepMemoryMode {
  return value === 'off' || value === 'smart' || value === 'always' ? value : 'always'
}

export function DeepMemoryModeField({
  onChange,
  value
}: {
  onChange: (value: DeepMemoryMode) => void
  value: unknown
}) {
  const { t } = useI18n()
  const c = t.settings.config
  const mode = normalizeDeepMemoryMode(value)
  const descriptions: Record<DeepMemoryMode, string> = {
    off: c.deepMemoryModeOffDesc,
    smart: c.deepMemoryModeSmartDesc,
    always: c.deepMemoryModeAlwaysDesc
  }

  return (
    <ListRow
      action={
        <SegmentedControl
          onChange={onChange}
          options={[
            { id: 'off', label: c.deepMemoryModeOff },
            { id: 'smart', label: c.deepMemoryModeSmart },
            { id: 'always', label: c.deepMemoryModeAlways }
          ]}
          value={mode}
        />
      }
      description={descriptions[mode]}
      title={c.deepMemoryModeTitle}
    />
  )
}
