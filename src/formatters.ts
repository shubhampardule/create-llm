export function formatParameterCount(parameters: number): string {
  const millions = parameters / 1_000_000;
  const formatted = millions < 10 ? millions.toFixed(1) : millions.toFixed(0);
  return `${formatted.replace(/\.0$/, '')}M`;
}
