const assert = require('node:assert/strict')
const test = require('node:test')
const vm = require('node:vm')
const launcher = require('../pinokio.js')

function evaluate(value, context) {
  if (typeof value !== 'string' || !value.startsWith('{{')) return value
  return vm.runInNewContext(value.slice(2, -2), context)
}

function steps(script, context) {
  const result = []
  for (const step of script.run) {
    if (step.when && !evaluate(step.when, context)) continue
    result.push(step)
    if (step.next === null) break
  }
  return result
}

test('each supported platform selects one torch backend without Windows true commands', () => {
  for (const platform of ['win32', 'linux', 'darwin']) {
    for (const gpu of ['nvidia', 'amd', 'cpu']) {
      for (const arch of ['x64', 'arm64']) {
        const context = { platform, gpu, arch, args: { path: 'app', venv: 'env' } }
        const selected = steps(require('../torch.js'), context)
        assert.equal(selected.length, 1, JSON.stringify(context))
        for (const command of [].concat(selected[0].params.message)) {
          assert.notEqual(evaluate(command, context), 'true')
        }
      }
    }
  }
})

test('all install branches reach verification before marking the environment ready', () => {
  for (const gpu of ['nvidia', 'amd', 'cpu']) {
    const selected = steps(require('../install.js'), { gpu, exists: () => true })
    assert.equal(selected[0].method, 'fs.rm')
    assert.equal(selected[1].params.uri, 'torch.js')
    const builds = selected.filter(s => typeof s.params.message === 'string' && s.params.message.includes('llama-cpp-python'))
    assert.equal(builds.length, 1)
    if (gpu === 'nvidia') assert.match(builds[0].params.message, /--no-binary llama-cpp-python/)
    assert.match(selected.at(-2).params.message[0], /^\(uv pip check \|\| .*Error: .*\)$/)
    assert.equal(selected.at(-1).params.path, 'app/env/.installed')
  }
})

test('update shares the complete installation workflow', () => {
  const update = require('../update.js').run
  assert.equal(update[0].params.message, 'git pull --ff-only')
  assert.equal(update[1].params.uri, 'install.js')
})

test('partial install and maintenance states never offer Start', async () => {
  for (const running of [null, 'install.js', 'reset.js', 'update.js', 'link.js']) {
    const menu = await launcher.menu(null, {
      exists: p => p === 'app/env',
      running: p => p === running,
      local: () => null
    })
    assert.equal(menu[0].href, running || 'install.js')
    assert.equal(menu[0].default, true)
  }
})

test('ready and running environments open the correct destination', async () => {
  for (const running of [false, true]) {
    const menu = await launcher.menu(null, {
      exists: () => true,
      running: p => running && p === 'start.js',
      local: () => ({ url: 'http://127.0.0.1:12345' })
    })
    assert.equal(menu[0].href, running ? 'http://127.0.0.1:12345' : 'start.js')
  }
})

test('server URL capture exposes the captured group to the menu', () => {
  const start = require('../start.js')
  const event = start.run[0].params.on[0]
  const match = new RegExp(event.event.slice(1, -1)).exec('Running on local URL: http://127.0.0.1:7862')
  assert.equal(match[1], 'http://127.0.0.1:7862')
  assert.equal(start.daemon, true)
  assert.equal(event.done, true)
  assert.equal(start.run[1].params.url, '{{input.event[1]}}')
})
