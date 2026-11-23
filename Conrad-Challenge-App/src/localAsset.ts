import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';

export async function readLocalAsset(moduleId: number, encoding: 'base64' | 'text') {
  const asset = Asset.fromModule(moduleId);
  await asset.downloadAsync();
  if (!asset.localUri) throw new Error('A bundled asset was unavailable locally');
  const file = new File(asset.localUri);
  return encoding === 'base64' ? file.base64() : file.text();
}
