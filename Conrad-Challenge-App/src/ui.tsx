import { useEffect, useRef, useState } from 'react';
import type { ComponentProps, ReactNode } from 'react';
import { Animated, Easing, Image, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
import type { StyleProp, TextInputProps, ViewStyle } from 'react-native';

// The app's original palette (the one the logo in assets/icon.png is drawn in). Status colours extend it:
// mint for clear, the original amber for a warning, the original danger peach for an alert, and one deeper
// red for the critical step only.
export const colors = {
  bg: '#09141C',
  surface: '#12232D',
  raised: '#1B3340',
  elevated: '#22404F',
  line: '#34505D',
  lineSoft: '#1F3A47',
  text: '#F5F9F9',
  muted: '#B3C5CC',
  faint: '#7A929C',
  brand: '#9BE7CB',
  brandDeep: '#5FC7A3',
  info: '#9BE7CB',
  safe: '#9BE7CB',
  watch: '#FFD48A',
  alert: '#FFC1A8',
  critical: '#FF7A6B',
  // Older names kept for modules that still import them.
  accent: '#9BE7CB',
  amber: '#FFD48A',
  danger: '#FFC1A8',
};

const NATIVE_DRIVER = Platform.OS !== 'web';
const LOGO = require('../assets/icon.png');

/** The app icon from assets/icon.png, used as the logo. */
export function Logo({ size = 30 }: { size?: number }) {
  return <Image source={LOGO} accessibilityLabel="Driven" style={{ width: size, height: size, borderRadius: size * 0.26 }} />;
}

/** Fades and lifts its children in when mounted; `delay` staggers lists. */
export function FadeIn({ children, delay = 0, style }: { children: ReactNode; delay?: number; style?: StyleProp<ViewStyle> }) {
  const progress = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(progress, { toValue: 1, duration: 380, delay, easing: Easing.out(Easing.cubic), useNativeDriver: NATIVE_DRIVER }).start();
  }, [delay, progress]);
  return <Animated.View style={[style, { opacity: progress,
    transform: [{ translateY: progress.interpolate({ inputRange: [0, 1], outputRange: [14, 0] }) }] }]}>{children}</Animated.View>;
}

/** Pressable that springs down slightly while held. */
export function PressScale({ children, onPress, disabled = false, style, accessibilityLabel, accessibilityRole = 'button' }: {
  children: ReactNode; onPress?: () => void; disabled?: boolean; style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string; accessibilityRole?: 'button' | 'switch' | 'tab';
}) {
  const scale = useRef(new Animated.Value(1)).current;
  const to = (value: number) => Animated.spring(scale, { toValue: value, speed: 40, bounciness: value === 1 ? 8 : 0, useNativeDriver: NATIVE_DRIVER }).start();
  return <Pressable accessibilityRole={accessibilityRole} accessibilityLabel={accessibilityLabel} disabled={disabled || !onPress}
    onPress={onPress} onPressIn={() => to(0.96)} onPressOut={() => to(1)}>
    <Animated.View style={[style, { transform: [{ scale }] }, disabled && { opacity: 0.45 }]}>{children}</Animated.View>
  </Pressable>;
}

/** A number that counts to its new value. */
export function useAnimatedNumber(target: number, duration = 450) {
  const [shown, setShown] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = Date.now();
    const begin = from.current;
    let frame = 0;
    const step = () => {
      const t = Math.min(1, (Date.now() - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = begin + (target - begin) * eased;
      from.current = value;
      setShown(value);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);
  return shown;
}

export type Tone = 'brand' | 'info' | 'safe' | 'watch' | 'alert' | 'critical' | 'off';

export function toneColor(tone: Tone) {
  switch (tone) {
    case 'brand': return colors.brand;
    case 'info': return colors.info;
    case 'safe': return colors.safe;
    case 'watch': return colors.watch;
    case 'alert': return colors.alert;
    case 'critical': return colors.critical;
    case 'off': return colors.faint;
  }
}

/** Hex colour with an alpha byte, for tinted fills and rings. */
export function tint(hex: string, alpha: number) {
  const byte = Math.round(Math.min(1, Math.max(0, alpha)) * 255).toString(16).padStart(2, '0');
  return hex + byte;
}

export function Heading({ children, size = 30 }: { children: ReactNode; size?: number }) {
  return <Text accessibilityRole="header" style={[styles.heading, { fontSize: size, lineHeight: Math.round(size * 1.15) }]}>{children}</Text>;
}

export function Eyebrow({ children, tone = 'off' }: { children: ReactNode; tone?: Tone }) {
  return <Text style={[styles.eyebrow, { color: tone === 'off' ? colors.muted : toneColor(tone) }]}>{children}</Text>;
}

export function Label({ children }: { children: ReactNode }) {
  return <Text style={styles.label}>{children}</Text>;
}

export function Body({ children, urgent = false, small = false }: { children: ReactNode; urgent?: boolean; small?: boolean }) {
  return <Text accessibilityLiveRegion={urgent ? 'polite' : undefined}
    style={[styles.body, small && styles.small, urgent && { color: colors.watch }]}>{children}</Text>;
}

export function ScreenTitle({ eyebrow, title, subtitle, right }: {
  eyebrow?: string; title: string; subtitle?: string; right?: ReactNode;
}) {
  return <View style={styles.screenTitle}>
    <View style={{ flex: 1, gap: 6 }}>
      {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
      <Heading>{title}</Heading>
      {subtitle ? <Body>{subtitle}</Body> : null}
    </View>
    {right}
  </View>;
}

export function Card({ children, tone, style, flat = false }: {
  children: ReactNode; tone?: Tone; style?: StyleProp<ViewStyle>; flat?: boolean;
}) {
  const accent = tone ? toneColor(tone) : null;
  return <View style={[styles.card, flat && styles.cardFlat,
    accent ? { borderColor: tint(accent, 0.4), backgroundColor: tint(accent, 0.06) } : null, style]}>{children}</View>;
}

export function SectionHeader({ title, action }: { title: string; action?: ReactNode }) {
  return <View style={styles.sectionHeader}>
    <Text style={styles.sectionTitle}>{title}</Text>
    {action}
  </View>;
}

export function Pill({ label, tone = 'brand', solid = false }: { label: string; tone?: Tone; solid?: boolean }) {
  const color = toneColor(tone);
  return <View style={[styles.pill, solid
    ? { backgroundColor: color, borderColor: color }
    : { backgroundColor: tint(color, 0.12), borderColor: tint(color, 0.35) }]}>
    {!solid && <View style={[styles.dot, { backgroundColor: color }]} />}
    <Text numberOfLines={1} style={[styles.pillText, { color: solid ? colors.bg : color }]}>{label}</Text>
  </View>;
}

export function LiveDot({ color = colors.critical, label = 'LIVE' }: { color?: string; label?: string }) {
  const pulse = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const loop = Animated.loop(Animated.timing(pulse, { toValue: 1, duration: 1400, easing: Easing.out(Easing.quad), useNativeDriver: NATIVE_DRIVER }));
    loop.start();
    return () => loop.stop();
  }, [pulse]);
  return <View style={styles.live}>
    <View style={{ width: 8, height: 8 }}>
      <Animated.View style={[StyleSheet.absoluteFill, { borderRadius: 4, backgroundColor: color,
        opacity: pulse.interpolate({ inputRange: [0, 1], outputRange: [0.6, 0] }),
        transform: [{ scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 2.8] }) }] }]} />
      <View style={[styles.liveDot, { backgroundColor: color }]} />
    </View>
    {label ? <Text style={[styles.liveText, { color }]}>{label}</Text> : null}
  </View>;
}

export function Stat({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: Tone }) {
  return <View style={styles.stat}>
    <Text style={[styles.statValue, tone ? { color: toneColor(tone) } : null]} numberOfLines={1}>
      {value}{unit ? <Text style={styles.statUnit}> {unit}</Text> : null}
    </Text>
    <Text style={styles.statLabel} numberOfLines={1}>{label}</Text>
  </View>;
}

/** Segmented risk meter, 0..1. Segments past the score are dim; lit segments shade from green to red. */
export function RiskMeter({ score, segments = 28, height = 26 }: { score: number; segments?: number; height?: number }) {
  const lit = Math.round(Math.min(1, Math.max(0, score)) * segments);
  return <View style={[styles.meter, { height }]} accessibilityRole="progressbar"
    accessibilityValue={{ min: 0, max: 100, now: Math.round(score * 100) }}>
    {Array.from({ length: segments }, (_, i) => {
      const at = i / segments;
      const color = at < 0.35 ? colors.safe : at < 0.6 ? colors.watch : at < 0.8 ? colors.alert : colors.critical;
      return <View key={i} style={[styles.segment, { backgroundColor: i < lit ? color : colors.elevated }]} />;
    })}
  </View>;
}

export function Avatar({ initials, color, size = 44, ring }: { initials: string; color: string; size?: number; ring?: string }) {
  return <View style={{ width: size + 8, height: size + 8, borderRadius: size, borderWidth: 2,
    borderColor: ring ?? 'transparent', alignItems: 'center', justifyContent: 'center' }}>
    <View style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: color, alignItems: 'center', justifyContent: 'center' }}>
      <Text style={{ color: colors.bg, fontWeight: '900', fontSize: size * 0.36, letterSpacing: 0.5 }}>{initials}</Text>
    </View>
  </View>;
}

/** Kept for screens that still use it: a single status disc with title and caption. */
export function StatusOrb({ tone, title, caption, size = 150 }: { tone: Tone; title: string; caption?: string; size?: number }) {
  const color = toneColor(tone);
  return <View style={styles.orbWrap} accessibilityLabel={`${title}. ${caption ?? ''}`}>
    <View style={{ width: size, height: size, borderRadius: size / 2, borderWidth: 2, borderColor: tint(color, 0.5),
      backgroundColor: tint(color, 0.08), alignItems: 'center', justifyContent: 'center' }}>
      <View style={{ width: size * 0.3, height: size * 0.3, borderRadius: size, backgroundColor: color }} />
    </View>
    <Text style={styles.orbTitle}>{title}</Text>
    {caption ? <Text style={styles.orbCaption}>{caption}</Text> : null}
  </View>;
}

export function Action({ title, onPress, secondary = false, disabled = false, tone, compact = false, style }: {
  title: string; onPress: () => void; secondary?: boolean; disabled?: boolean; tone?: 'danger' | 'ghost' | 'critical';
  compact?: boolean; style?: StyleProp<ViewStyle>;
}) {
  const ghost = tone === 'ghost';
  const danger = tone === 'danger';
  const critical = tone === 'critical';
  return (
    <PressScale onPress={onPress} disabled={disabled} accessibilityLabel={title}
      style={[styles.button, compact && styles.buttonCompact,
        secondary && styles.secondary, ghost && styles.ghost, danger && styles.dangerButton, critical && styles.criticalButton, style]}>
      <Text style={[styles.buttonText, (secondary || ghost) && { color: colors.text },
        danger && { color: colors.alert }, critical && { color: colors.bg }, compact && { fontSize: 14 }]}>{title}</Text>
    </PressScale>
  );
}

export function Chip({ label, selected, onPress, disabled = false }: {
  label: string; selected: boolean; onPress: () => void; disabled?: boolean;
}) {
  return <Pressable accessibilityRole="button" accessibilityState={{ selected, disabled }} disabled={disabled}
    onPress={onPress} style={[styles.chip, selected && styles.chipSelected]}>
    <Text numberOfLines={1} style={[styles.chipText, selected && { color: colors.bg }]}>{label}</Text>
  </Pressable>;
}

export function Field({ label, ...props }: TextInputProps & { label: string }) {
  return (
    <View style={{ gap: 6 }}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput {...props} accessibilityLabel={label} placeholderTextColor={colors.faint} style={styles.input} />
    </View>
  );
}

export function Divider() {
  return <View style={styles.divider} />;
}

export type IconName = ComponentProps<typeof MaterialCommunityIcons>['name'];

export function Icon({ name, size = 20, color = colors.text }: { name: IconName; size?: number; color?: string }) {
  return <MaterialCommunityIcons name={name} size={size} color={color} />;
}

export function TabBar<T extends string>({ items, active, onChange, disabled = false, icons }: {
  items: readonly T[]; active: T; onChange: (item: T) => void; disabled?: boolean; icons?: Record<T, IconName>;
}) {
  return <View style={styles.tabs} accessibilityRole="tablist">
    {items.map((item) => {
      const selected = item === active;
      const color = selected ? colors.text : colors.faint;
      return <Pressable key={item} style={styles.tab} disabled={disabled}
        onPress={() => onChange(item)} accessibilityRole="tab" accessibilityState={{ selected }}>
        <TabIcon selected={selected} name={icons?.[item]} />
        <Text numberOfLines={1} maxFontSizeMultiplier={1.2} style={[styles.tabText, { color }]}>{item}</Text>
      </Pressable>;
    })}
  </View>;
}

function TabIcon({ selected, name }: { selected: boolean; name?: IconName }) {
  const on = useRef(new Animated.Value(selected ? 1 : 0)).current;
  useEffect(() => {
    Animated.spring(on, { toValue: selected ? 1 : 0, speed: 18, bounciness: 10, useNativeDriver: NATIVE_DRIVER }).start();
  }, [on, selected]);
  if (!name) return null;
  return <Animated.View style={{ alignItems: 'center', justifyContent: 'center', width: 56, height: 30, borderRadius: 15,
    transform: [{ scale: on.interpolate({ inputRange: [0, 1], outputRange: [1, 1.08] }) }] }}>
    <Animated.View style={[StyleSheet.absoluteFill, { borderRadius: 15, backgroundColor: tint(colors.brand, 0.18), opacity: on }]} />
    <Icon name={name} size={22} color={selected ? colors.brand : colors.faint} />
  </Animated.View>;
}

/** Round control with an icon and a caption, Tesla quick-control style. */
export function IconButton({ icon, label, active = false, disabled = false, onPress, tone }: {
  icon: IconName; label: string; active?: boolean; disabled?: boolean; onPress: () => void; tone?: Tone;
}) {
  const color = tone ? toneColor(tone) : colors.brand;
  return <PressScale accessibilityRole="switch" accessibilityLabel={label} disabled={disabled} onPress={onPress} style={styles.iconButtonWrap}>
    <View style={[styles.iconButton, active && { backgroundColor: color, borderColor: color }]}>
      <Icon name={icon} size={26} color={active ? colors.bg : colors.text} />
    </View>
    <Text style={[styles.iconLabel, active && { color: colors.text }]} numberOfLines={1}>{label}</Text>
  </PressScale>;
}

/** Settings-style row: icon tile, title and subtitle, a value on the right and a chevron. */
export function ListRow({ icon, title, subtitle, value, valueTone, onPress, tone, last = false }: {
  icon: IconName; title: string; subtitle?: string; value?: string; valueTone?: Tone; onPress?: () => void; tone?: Tone; last?: boolean;
}) {
  const accent = tone ? toneColor(tone) : colors.muted;
  return <Pressable accessibilityRole={onPress ? 'button' : undefined} disabled={!onPress} onPress={onPress}
    style={({ pressed }) => [styles.listRow, !last && styles.listRowLine, pressed && { backgroundColor: colors.raised }]}>
    <View style={[styles.listIcon, { backgroundColor: tint(accent, 0.14) }]}><Icon name={icon} size={18} color={accent} /></View>
    <View style={{ flex: 1, gap: 2 }}>
      <Text style={styles.listTitle} numberOfLines={1}>{title}</Text>
      {subtitle ? <Text style={styles.listSubtitle} numberOfLines={2}>{subtitle}</Text> : null}
    </View>
    {value ? <Text style={[styles.listValue, valueTone ? { color: toneColor(valueTone) } : null]} numberOfLines={1}>{value}</Text> : null}
    {onPress ? <Icon name="chevron-right" size={22} color={colors.faint} /> : null}
  </Pressable>;
}

/** Grouped list container. */
export function ListGroup({ children }: { children: ReactNode }) {
  return <View style={styles.listGroup}>{children}</View>;
}

/** Thin continuous bar, 0..1, coloured by level. */
export function Gauge({ value, height = 6 }: { value: number; height?: number }) {
  const v = Math.min(1, Math.max(0, value));
  const color = v < 0.35 ? colors.safe : v < 0.6 ? colors.watch : v < 0.8 ? colors.alert : colors.critical;
  const width = useRef(new Animated.Value(v)).current;
  useEffect(() => {
    Animated.timing(width, { toValue: v, duration: 500, easing: Easing.out(Easing.cubic), useNativeDriver: false }).start();
  }, [v, width]);
  return <View style={{ height, borderRadius: height, backgroundColor: colors.elevated, overflow: 'hidden' }}
    accessibilityRole="progressbar" accessibilityValue={{ min: 0, max: 100, now: Math.round(v * 100) }}>
    <Animated.View style={{ height, borderRadius: height, backgroundColor: color,
      width: width.interpolate({ inputRange: [0, 1], outputRange: ['2%', '100%'] }) }} />
  </View>;
}

/** Large light numeral with a small caption, for hero readouts. */
export function Metric({ value, unit, label, tone, size = 44 }: { value: string; unit?: string; label: string; tone?: Tone; size?: number }) {
  const numeric = Number(value);
  const counted = useAnimatedNumber(Number.isFinite(numeric) ? numeric : 0);
  const shown = Number.isFinite(numeric) && value.trim() !== '' ? String(Math.round(counted)) : value;
  return <View style={{ gap: 2, alignItems: 'flex-end' }}>
    <Text style={[styles.metric, { fontSize: size, lineHeight: size * 1.1 }, tone ? { color: toneColor(tone) } : null]}>
      {shown}{unit ? <Text style={styles.metricUnit}> {unit}</Text> : null}
    </Text>
    <Text style={styles.metricLabel}>{label}</Text>
  </View>;
}

/** Four-up status strip: icon, value, label. */
export function StatusStrip({ items }: { items: { icon: IconName; value: string; label: string; tone?: Tone }[] }) {
  return <View style={styles.strip}>
    {items.map((item, index) => <View key={item.label} style={[styles.stripItem, index > 0 && styles.stripDivider]}>
      <Icon name={item.icon} size={20} color={item.tone ? toneColor(item.tone) : colors.muted} />
      <Text style={[styles.stripValue, item.tone && item.tone !== 'safe' ? { color: toneColor(item.tone) } : null]} numberOfLines={1}>{item.value}</Text>
      <Text style={styles.stripLabel} numberOfLines={1}>{item.label}</Text>
    </View>)}
  </View>;
}

export const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.bg },
  content: { width: '100%', maxWidth: 640, alignSelf: 'center', paddingHorizontal: 18, paddingTop: 8, paddingBottom: 28, gap: 14 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 18, paddingVertical: 12, backgroundColor: colors.bg },
  brand: { color: colors.text, fontSize: 17, fontWeight: '700', letterSpacing: 0.2 },
  brandMark: { width: 24, height: 24, borderRadius: 8, backgroundColor: colors.brand, alignItems: 'center', justifyContent: 'center' },
  brandMarkInner: { width: 10, height: 4, borderRadius: 2, backgroundColor: colors.bg },
  heading: { color: colors.text, fontWeight: '600', letterSpacing: -0.6 },
  eyebrow: { fontSize: 11, fontWeight: '800', letterSpacing: 1.6, textTransform: 'uppercase' },
  label: { color: colors.text, fontSize: 16, fontWeight: '800', letterSpacing: -0.2 },
  body: { color: colors.muted, fontSize: 15, lineHeight: 22 },
  small: { fontSize: 13, lineHeight: 18 },
  screenTitle: { flexDirection: 'row', alignItems: 'flex-end', gap: 12, paddingTop: 4, paddingBottom: 2 },
  card: { padding: 18, gap: 12, backgroundColor: colors.surface, borderRadius: 24, borderWidth: 1, borderColor: colors.lineSoft },
  cardFlat: { backgroundColor: colors.surface, borderRadius: 18, padding: 14 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 10 },
  sectionTitle: { color: colors.muted, fontSize: 12, fontWeight: '800', letterSpacing: 1.4, textTransform: 'uppercase' },
  pill: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', paddingHorizontal: 10,
    paddingVertical: 5, borderRadius: 999, borderWidth: 1 },
  pillText: { fontSize: 12, fontWeight: '800', letterSpacing: 0.2 },
  dot: { width: 7, height: 7, borderRadius: 4 },
  live: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  liveDot: { width: 8, height: 8, borderRadius: 4 },
  liveText: { fontSize: 11, fontWeight: '900', letterSpacing: 1.4 },
  stat: { flex: 1, minWidth: 90, gap: 2, paddingVertical: 14, paddingHorizontal: 14, borderRadius: 18, backgroundColor: colors.surface,
    borderWidth: 1, borderColor: colors.lineSoft },
  statLabel: { color: colors.muted, fontSize: 12, fontWeight: '700' },
  statValue: { color: colors.text, fontSize: 26, fontWeight: '900', letterSpacing: -0.8, fontVariant: ['tabular-nums'] },
  statUnit: { color: colors.muted, fontSize: 13, fontWeight: '800' },
  meter: { flexDirection: 'row', gap: 3, alignItems: 'stretch' },
  segment: { flex: 1, borderRadius: 3 },
  orbWrap: { alignItems: 'center', gap: 10, paddingVertical: 6 },
  orbTitle: { color: colors.text, fontSize: 24, fontWeight: '900', letterSpacing: -0.5, textAlign: 'center' },
  orbCaption: { color: colors.muted, fontSize: 15, lineHeight: 21, textAlign: 'center', maxWidth: 320 },
  button: { minHeight: 56, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.brand,
    paddingHorizontal: 18, paddingVertical: 12, borderRadius: 18, overflow: 'hidden' },
  buttonCompact: { minHeight: 42, borderRadius: 14, paddingVertical: 8, paddingHorizontal: 14 },
  secondary: { backgroundColor: colors.raised, borderWidth: 1, borderColor: colors.line },
  ghost: { backgroundColor: 'transparent', borderWidth: 1, borderColor: colors.line },
  dangerButton: { backgroundColor: tint(colors.alert, 0.08), borderWidth: 1, borderColor: tint(colors.alert, 0.45) },
  criticalButton: { backgroundColor: colors.critical },
  buttonText: { color: colors.bg, fontWeight: '800', fontSize: 16, textAlign: 'center', letterSpacing: 0.1 },
  chip: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999, borderWidth: 1, borderColor: colors.line,
    backgroundColor: colors.raised, maxWidth: '100%' },
  chipSelected: { borderColor: colors.brand, backgroundColor: colors.brand },
  chipText: { color: colors.muted, fontWeight: '800', fontSize: 14 },
  fieldLabel: { color: colors.muted, fontSize: 13, fontWeight: '700' },
  input: { minHeight: 52, borderRadius: 16, borderWidth: 1, borderColor: colors.line, color: colors.text,
    backgroundColor: colors.bg, paddingHorizontal: 14, paddingVertical: 10, fontSize: 16 },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 },
  wrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  divider: { height: StyleSheet.hairlineWidth, backgroundColor: colors.line },
  tabs: { flexDirection: 'row', paddingTop: 8, paddingBottom: 6, backgroundColor: colors.bg,
    borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  tab: { flex: 1, minHeight: 50, alignItems: 'center', justifyContent: 'center', gap: 3 },
  tabText: { fontSize: 11, fontWeight: '600', letterSpacing: 0.2 },
  iconButtonWrap: { alignItems: 'center', gap: 8, flex: 1 },
  iconButton: { width: 58, height: 58, borderRadius: 29, alignItems: 'center', justifyContent: 'center',
    backgroundColor: colors.raised, borderWidth: 1, borderColor: colors.line },
  iconLabel: { color: colors.muted, fontSize: 12, fontWeight: '600' },
  listGroup: { backgroundColor: colors.surface, borderRadius: 20, overflow: 'hidden', borderWidth: 1, borderColor: colors.lineSoft },
  listRow: { flexDirection: 'row', alignItems: 'center', gap: 14, paddingHorizontal: 16, paddingVertical: 14, minHeight: 64 },
  listRowLine: { borderBottomWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  listIcon: { width: 36, height: 36, borderRadius: 11, alignItems: 'center', justifyContent: 'center' },
  listTitle: { color: colors.text, fontSize: 16, fontWeight: '600' },
  listSubtitle: { color: colors.muted, fontSize: 13, lineHeight: 18 },
  listValue: { color: colors.muted, fontSize: 14, fontWeight: '600', maxWidth: 118, textAlign: 'right' },
  metric: { color: colors.text, fontWeight: '300', letterSpacing: -1.5, fontVariant: ['tabular-nums'] },
  metricUnit: { color: colors.muted, fontSize: 16, fontWeight: '500', letterSpacing: 0 },
  metricLabel: { color: colors.muted, fontSize: 12, fontWeight: '600', letterSpacing: 1.2, textTransform: 'uppercase' },
  strip: { flexDirection: 'row', backgroundColor: colors.surface, borderRadius: 20, paddingVertical: 14, borderWidth: 1, borderColor: colors.lineSoft },
  stripItem: { flex: 1, alignItems: 'center', gap: 4, paddingHorizontal: 4 },
  stripDivider: { borderLeftWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  stripValue: { color: colors.text, fontSize: 16, fontWeight: '600' },
  stripLabel: { color: colors.faint, fontSize: 11, fontWeight: '600', letterSpacing: 0.4 },
  sessionFooter: { paddingHorizontal: 18, paddingVertical: 12, gap: 10, backgroundColor: colors.bg },
});
