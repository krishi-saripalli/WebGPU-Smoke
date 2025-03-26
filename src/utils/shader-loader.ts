export async function loadShader(path: string): Promise<string> {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const text = await response.text();
    if (!text) {
      throw new Error('Shader file is empty');
    }
    return text;
  } catch (error) {
    console.error(`Failed to load shader from ${path}:`, error);
    throw error;
  }
}
