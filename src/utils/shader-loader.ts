export async function loadShader(name: string): Promise<string> {
  const response = await fetch(`${import.meta.env.BASE_URL}shaders/${name}.wgsl`);
  if (!response.ok) throw new Error(`Unable to load ${name}.wgsl (HTTP ${response.status})`);
  return response.text();
}

export async function loadShaderModule(device: GPUDevice, name: string, common: string) {
  const source = await loadShader(name);
  const shader = device.createShaderModule({
    label: name,
    code: source.replace('@import "common.wgsl";', common),
  });
  const info = await shader.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length) {
    throw new Error(errors.map((e) => `${name}:${e.lineNum} ${e.message}`).join('\n'));
  }
  return shader;
}
