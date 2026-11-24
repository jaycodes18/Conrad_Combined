import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Animated, Easing, Platform, StyleSheet, Text, View } from 'react-native';
import type { LiveFace, TileTone } from '../liveView';
import type { Tier } from '../risk';
import { colors, tint, toneColor } from '../ui';

type TileLike = { key?: string; title: string; value: string; note: string; tone: TileTone };

export function IndicatorGrid({ tiles }: { tiles: TileLike[] }) {
  return <View style={parts.grid}>
    {tiles.map((tile) => {
      const color = toneColor(tile.tone);
      const off = tile.tone === 'off';
      const hot = tile.tone === 'alert' || tile.tone === 'critical';
      return <View key={tile.key ?? tile.title} style={[parts.tile, hot && { borderColor: tint(color, 0.55), backgroundColor: tint(color, 0.08) }]}
        accessibilityLabel={`${tile.title}: ${tile.value}. ${tile.note}`}>
        <View style={parts.tileHead}>
          <Text style={parts.tileTitle} numberOfLines={1}>{tile.title}</Text>
          <View style={[parts.tileDot, { backgroundColor: color }, off && { opacity: 0.5 }]} />
        </View>
        <Text style={[parts.tileValue, { color: off ? colors.faint : hot ? color : colors.text }]} numberOfLines={1}>{tile.value}</Text>
        <Text style={parts.tileNote} numberOfLines={2}>{tile.note}</Text>
      </View>;
    })}
  </View>;
}

const STEPS: { tier: Tier; title: string }[] = [
  { tier: 0, title: 'Watching' },
  { tier: 1, title: 'Driver alert' },
  { tier: 2, title: 'Family alert' },
  { tier: 3, title: 'Call step' },
];

/** Horizontal four-step escalation track. */
export function EscalationLadder({ tier }: { tier: Tier }) {
  return <View style={parts.track} accessibilityLabel={`Escalation step ${tier} of 3`}>
    {STEPS.map((step) => {
      const reached = tier >= step.tier;
      const color = step.tier === 0 ? colors.safe : step.tier === 1 ? colors.watch : step.tier === 2 ? colors.alert : colors.critical;
      return <View key={step.tier} style={parts.trackStep}>
        <View style={[parts.trackBar, { backgroundColor: reached ? color : colors.elevated }]} />
        <Text style={[parts.trackText, { color: reached ? colors.text : colors.faint }]} numberOfLines={1}>{step.title}</Text>
      </View>;
    })}
  </View>;
}

const NATIVE_DRIVER = Platform.OS !== 'web';

/** Slides in, then breathes while it is shown, so a driver glancing at the phone notices it. */
export function Banner({ text, sub, tone }: { text: string; sub?: string; tone: 'critical' | 'watch' | 'brand' }) {
  const color = toneColor(tone);
  const enter = useRef(new Animated.Value(0)).current;
  const pulse = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.spring(enter, { toValue: 1, speed: 14, bounciness: 9, useNativeDriver: NATIVE_DRIVER }).start();
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(pulse, { toValue: 1, duration: 600, easing: Easing.inOut(Easing.quad), useNativeDriver: NATIVE_DRIVER }),
      Animated.timing(pulse, { toValue: 0, duration: 600, easing: Easing.inOut(Easing.quad), useNativeDriver: NATIVE_DRIVER }),
    ]));
    loop.start();
    return () => loop.stop();
  }, [enter, pulse]);
  return <Animated.View style={[parts.banner, { backgroundColor: tone === 'critical' ? color : tint(color, 0.14), borderColor: color,
    opacity: enter, transform: [
      { translateY: enter.interpolate({ inputRange: [0, 1], outputRange: [-16, 0] }) },
      { scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.025] }) },
    ] }]} accessibilityLiveRegion="assertive">
    <Text style={[parts.bannerText, { color: tone === 'critical' ? colors.bg : color }]}>{text}</Text>
    {sub ? <Text style={[parts.bannerSub, { color: tone === 'critical' ? tint(colors.bg, 0.8) : colors.muted }]}>{sub}</Text> : null}
  </Animated.View>;
}

/** One eye that animates between open and shut. */
function Eye({ closed, color, yaw }: { closed: boolean; color: string; yaw: number }) {
  const open = useRef(new Animated.Value(closed ? 0 : 1)).current;
  useEffect(() => {
    Animated.timing(open, { toValue: closed ? 0 : 1, duration: 180, easing: Easing.out(Easing.quad), useNativeDriver: false }).start();
  }, [closed, open]);
  return <Animated.View style={[parts.eye, { borderColor: color, height: open.interpolate({ inputRange: [0, 1], outputRange: [3, 13] }),
    backgroundColor: open.interpolate({ inputRange: [0, 1], outputRange: [color, 'rgba(0,0,0,0)'] }) }]}>
    <Animated.View style={[parts.pupil, { backgroundColor: color, opacity: open, transform: [{ translateX: yaw * 0.12 }] }]} />
  </Animated.View>;
}

/** Corner brackets and a face oval, drawn over a camera preview to help framing. */
export function FaceGuide({ color = '#FFFFFF', children }: { color?: string; children?: ReactNode }) {
  return <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
    <View style={[parts.oval, { borderColor: tint(color, 0.55) }]} pointerEvents="none" />
    {children}
  </View>;
}

/** Pill drawn on top of the camera image. */
export function HudChip({ children, color = colors.text }: { children: ReactNode; color?: string }) {
  return <View style={parts.hud}><Text style={[parts.hudText, { color }]} numberOfLines={1}>{children}</Text></View>;
}

/**
 * The parent's live view. Video never leaves the teen's phone, so this draws what the model reads from
 * the face: eyes open or closed and where the head points.
 */
export function SensorFace({ face, tone, height = 230, compact = false }: { face: LiveFace | null; tone: TileTone; height?: number; compact?: boolean }) {
  const color = tone === 'off' ? colors.faint : toneColor(tone);
  const yaw = Math.max(-40, Math.min(40, face?.yaw ?? 0));
  const pitch = Math.max(-30, Math.min(30, face?.pitch ?? 0));
  const closed = face?.eyes === 'closed';
  const turn = useRef(new Animated.Value(yaw)).current;
  const tilt = useRef(new Animated.Value(pitch)).current;
  useEffect(() => {
    Animated.parallel([
      Animated.spring(turn, { toValue: yaw, speed: 10, bounciness: 4, useNativeDriver: NATIVE_DRIVER }),
      Animated.spring(tilt, { toValue: pitch, speed: 10, bounciness: 4, useNativeDriver: NATIVE_DRIVER }),
    ]).start();
  }, [yaw, pitch, turn, tilt]);
  const scale = compact ? 0.62 : 1;
  return <View style={[parts.sensor, { height }, compact && { borderRadius: 0, borderWidth: 0 }]}>
    {Array.from({ length: 9 }, (_, i) => <View key={'h' + i} style={[parts.gridH, { top: `${(i + 1) * 10}%` }]} />)}
    {Array.from({ length: 6 }, (_, i) => <View key={'v' + i} style={[parts.gridV, { left: `${(i + 1) * 14.3}%` }]} />)}
    {face ? <Animated.View style={[parts.head, { borderColor: color, shadowColor: color,
      transform: [
        { scale },
        { translateX: Animated.multiply(turn, 1.4) },
        { translateY: Animated.multiply(tilt, -1.2) },
        { rotate: turn.interpolate({ inputRange: [-40, 40], outputRange: ['-10deg', '10deg'] }) },
      ] }]}>
      <View style={parts.eyes}>
        {[0, 1].map((i) => <Eye key={i} closed={closed} color={color} yaw={yaw} />)}
      </View>
      <View style={[parts.nose, { backgroundColor: tint(color, 0.6) }]} />
      <View style={[parts.mouth, { backgroundColor: tint(color, 0.6) }]} />
    </Animated.View> : <Text style={parts.noFace}>Face not visible</Text>}
    {!compact && <FaceGuide color={color} />}
  </View>;
}

const parts = StyleSheet.create({
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  tile: { flexBasis: '47%', flexGrow: 1, minHeight: 100, padding: 14, gap: 5, borderRadius: 20,
    backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.lineSoft },
  tileHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  tileDot: { width: 8, height: 8, borderRadius: 4 },
  tileTitle: { color: colors.muted, fontSize: 11, fontWeight: '800', letterSpacing: 1.1, textTransform: 'uppercase', flex: 1 },
  tileValue: { fontSize: 20, fontWeight: '900', letterSpacing: -0.4 },
  tileNote: { color: colors.faint, fontSize: 12, lineHeight: 16 },
  track: { flexDirection: 'row', gap: 6 },
  trackStep: { flex: 1, gap: 7 },
  trackBar: { height: 6, borderRadius: 3 },
  trackText: { fontSize: 11, fontWeight: '800' },
  banner: { borderWidth: 1.5, borderRadius: 22, paddingVertical: 16, paddingHorizontal: 18, gap: 4 },
  bannerText: { fontSize: 22, fontWeight: '900', textAlign: 'center', letterSpacing: -0.4 },
  bannerSub: { fontSize: 14, fontWeight: '700', textAlign: 'center' },
  oval: { position: 'absolute', alignSelf: 'center', top: '18%', width: '52%', height: '58%', borderRadius: 999, borderWidth: 1.5, borderStyle: 'dashed' },
  hud: { paddingHorizontal: 11, paddingVertical: 6, borderRadius: 999, backgroundColor: 'rgba(7,9,13,0.72)' },
  hudText: { fontSize: 12, fontWeight: '900', letterSpacing: 0.3 },
  sensor: { borderRadius: 26, overflow: 'hidden', backgroundColor: '#0A0D12', alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: colors.lineSoft },
  gridH: { position: 'absolute', left: 0, right: 0, height: StyleSheet.hairlineWidth, backgroundColor: '#FFFFFF0D' },
  gridV: { position: 'absolute', top: 0, bottom: 0, width: StyleSheet.hairlineWidth, backgroundColor: '#FFFFFF0D' },
  head: { width: 112, height: 142, borderRadius: 60, borderWidth: 2.5, alignItems: 'center', paddingTop: 48, gap: 14,
    shadowOpacity: 0.6, shadowRadius: 16, shadowOffset: { width: 0, height: 0 } },
  eyes: { flexDirection: 'row', gap: 22, height: 14, alignItems: 'center' },
  eye: { width: 24, height: 13, borderRadius: 7, borderWidth: 2, alignItems: 'center', justifyContent: 'center' },
  pupil: { width: 6, height: 6, borderRadius: 3 },
  nose: { width: 2, height: 16, borderRadius: 1 },
  mouth: { width: 30, height: 2.5, borderRadius: 2 },
  noFace: { color: colors.faint, fontWeight: '800', fontSize: 14 },
});
