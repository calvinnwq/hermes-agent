import { describe, expect, it, vi } from 'vitest'

import { sessionCommands } from '../app/slash/commands/session.js'

const yoloCommand = sessionCommands.find(cmd => cmd.name === 'yolo')!

const guarded =
  <T>(fn: (r: T) => void) =>
  (r: null | T) => {
    if (r) {
      fn(r)
    }
  }

const buildCtx = (response: unknown) => {
  const rpc = vi.fn(() => Promise.resolve(response))
  const sys = vi.fn()

  const ctx = {
    gateway: { rpc },
    guarded,
    guardedErr: vi.fn(),
    sid: 'sid-1',
    transcript: { sys }
  }

  const run = async (arg: string) => {
    const result = yoloCommand.run(arg, ctx as never, 'yolo')
    await result
    await Promise.resolve()
  }

  return { ctx, rpc, run, sys }
}

describe('/yolo slash command', () => {
  it('preserves the bare session-local toggle', async () => {
    const { rpc, run, sys } = buildCtx({ key: 'yolo', value: '1' })

    await run('   ')

    expect(rpc).toHaveBeenCalledWith('config.set', { key: 'yolo', session_id: 'sid-1' })
    expect(sys).toHaveBeenCalledWith('yolo on')
  })

  it('preserves the exact bare OFF response', async () => {
    const { rpc, run, sys } = buildCtx({ key: 'yolo', value: '0' })

    await run('')

    expect(rpc).toHaveBeenCalledWith('config.set', { key: 'yolo', session_id: 'sid-1' })
    expect(sys).toHaveBeenCalledWith('yolo off')
  })

  it('reports effective status without mutating session YOLO', async () => {
    const { rpc, run, sys } = buildCtx({ key: 'yolo', value: '1' })

    await run(' StAtUs ')

    expect(rpc).toHaveBeenCalledOnce()
    expect(rpc).toHaveBeenCalledWith('config.get', { key: 'yolo', session_id: 'sid-1' })
    expect(rpc).not.toHaveBeenCalledWith('config.set', expect.anything())
    expect(sys).toHaveBeenCalledWith('YOLO approval bypass ON.')
  })

  it('reports effective OFF status without mutating session YOLO', async () => {
    const { rpc, run, sys } = buildCtx({ key: 'yolo', value: '0' })

    await run('status')

    expect(rpc).toHaveBeenCalledOnce()
    expect(rpc).toHaveBeenCalledWith('config.get', { key: 'yolo', session_id: 'sid-1' })
    expect(rpc).not.toHaveBeenCalledWith('config.set', expect.anything())
    expect(sys).toHaveBeenCalledWith('YOLO approval bypass OFF.')
  })

  it('prints usage for an unknown argument without calling the gateway', async () => {
    const { rpc, run, sys } = buildCtx({ key: 'yolo', value: '1' })

    await run('nope')

    expect(rpc).not.toHaveBeenCalled()
    expect(sys).toHaveBeenCalledWith('/yolo [status]')
  })
})
