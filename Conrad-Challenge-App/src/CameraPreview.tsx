import type { ReactNode } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { StyleProp, ViewStyle } from 'react-native';
import { CameraView } from 'expo-camera';
import { colors } from './ui';

/**
 * Framing preview for the teen's Drive screen. It only shows the image; the face model starts with the
 * session. Nothing is captured or saved here.
 */
export function CameraPreview({ granted, style, children }: { granted: boolean; style?: StyleProp<ViewStyle>; children?: ReactNode }) {
  return <View style={[preview.frame, style]}>
    {granted
      ? <CameraView style={StyleSheet.absoluteFill} facing="front" active />
      : <View style={preview.empty}><Text style={preview.emptyText}>Camera off</Text></View>}
    {children}
  </View>;
}

const preview = StyleSheet.create({
  frame: { width: '100%', aspectRatio: 3 / 4, borderRadius: 28, overflow: 'hidden', backgroundColor: '#0A0D12' },
  empty: { ...StyleSheet.absoluteFill, alignItems: 'center', justifyContent: 'center' },
  emptyText: { color: colors.faint, fontWeight: '800', fontSize: 15 },
});
