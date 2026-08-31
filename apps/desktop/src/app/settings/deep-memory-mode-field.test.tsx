import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DeepMemoryModeField, normalizeDeepMemoryMode } from './deep-memory-mode-field'

afterEach(cleanup)

describe('DeepMemoryModeField', () => {
  it('uses always for missing or future config values', () => {
    expect(normalizeDeepMemoryMode(undefined)).toBe('always')
    expect(normalizeDeepMemoryMode('future')).toBe('always')
  })

  it('renders three mutually exclusive modes and reports a selection', () => {
    const onChange = vi.fn()

    render(<DeepMemoryModeField onChange={onChange} value="smart" />)

    expect(screen.getByRole('button', { name: 'Smart' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText(/also calls memo_recall/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Off' }))

    expect(onChange).toHaveBeenCalledWith('off')
  })
})
