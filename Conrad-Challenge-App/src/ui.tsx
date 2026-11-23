import type { ReactNode } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import type { TextInputProps } from 'react-native';

export const colors = {
  bg: '#09141C',
  surface: '#12232D',
  raised: '#1B3340',
  line: '#34505D',
  text: '#F5F9F9',
  muted: '#B3C5CC',
  accent: '#9BE7CB',
  amber: '#FFD48A',
  danger: '#FFC1A8',
};

export function Heading({ children }: { children: ReactNode }) {
  return <Text accessibilityRole="header" style={styles.heading}>{children}</Text>;
}

export function Label({ children }: { children: ReactNode }) {
  return <Text style={styles.label}>{children}</Text>;
}

export function Body({ children, urgent = false }: { children: ReactNode; urgent?: boolean }) {
  return <Text accessibilityLiveRegion={urgent ? 'polite' : undefined} style={[styles.body, urgent && { color: colors.amber }]}>{children}</Text>;
}

export function Card({ children }: { children: ReactNode }) {
  return <View style={styles.card}>{children}</View>;
}

export function Action({ title, onPress, secondary = false, disabled = false }: {
  title: string; onPress: () => void; secondary?: boolean; disabled?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled }}
      android_ripple={{ color: secondary ? '#FFFFFF20' : '#09141C22' }}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [styles.button, secondary && styles.secondary, (disabled || pressed) && { opacity: 0.6 }]}
    >
      <Text style={[styles.buttonText, secondary && { color: colors.text }]}>{title}</Text>
    </Pressable>
  );
}

export function Field({ label, ...props }: TextInputProps & { label: string }) {
  return (
    <View style={{ gap: 6 }}>
      <Label>{label}</Label>
      <TextInput
        {...props}
        accessibilityLabel={label}
        placeholderTextColor={colors.muted}
        style={styles.input}
      />
    </View>
  );
}

export const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: colors.bg },
  content: { width: '100%', maxWidth: 600, alignSelf: 'center', padding: 16, paddingBottom: 24, gap: 14 },
  header: { paddingHorizontal: 16, paddingVertical: 11, borderBottomWidth: 1, borderColor: colors.line, backgroundColor: colors.surface },
  brand: { color: colors.accent, fontSize: 13, fontWeight: '700', letterSpacing: 1.2 },
  heading: { color: colors.text, fontSize: 30, fontWeight: '800' },
  label: { color: colors.text, fontSize: 17, fontWeight: '700' },
  body: { color: colors.muted, fontSize: 15, lineHeight: 22 },
  card: { padding: 16, gap: 12, backgroundColor: colors.surface, borderRadius: 18, borderWidth: 1, borderColor: colors.line },
  button: { minHeight: 52, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.accent, paddingHorizontal: 14, paddingVertical: 10, borderRadius: 12, overflow: 'hidden' },
  secondary: { backgroundColor: colors.raised, borderWidth: 1, borderColor: colors.line },
  buttonText: { color: colors.bg, fontWeight: '800', fontSize: 15, textAlign: 'center' },
  input: { minHeight: 48, borderRadius: 8, borderWidth: 1, borderColor: colors.line, color: colors.text, backgroundColor: colors.bg, paddingHorizontal: 10, paddingVertical: 8, fontSize: 16 },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  tabs: { flexDirection: 'row', borderTopWidth: 1, borderColor: colors.line, backgroundColor: colors.surface, paddingHorizontal: 4 },
  tab: { flex: 1, minHeight: 52, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 2, overflow: 'hidden' },
  tabSelected: { backgroundColor: colors.raised },
  tabText: { color: colors.muted, fontSize: 12, fontWeight: '700', textAlign: 'center' },
  sessionFooter: { padding: 16, gap: 8, borderTopWidth: 1, borderColor: colors.line, backgroundColor: colors.surface },
});
