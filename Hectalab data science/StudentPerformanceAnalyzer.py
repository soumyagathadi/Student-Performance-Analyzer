import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class Config:
	marks_path: str = "marks.csv"
	attendance_path: str = "attendance.csv"
	demographics_path: str = "demographics.csv"
	dashboard_path: str = "student_analytics_dashboard.png"
	text_report_path: str = "student_analysis_summary.txt"
	excel_report_path: str = "student_analysis_detailed.xlsx"
	pdf_report_path: str = "student_analysis_report.pdf"
	pass_threshold: float = 40.0
	attendance_threshold: float = 70.0
	random_seed: int = 42
	synth_students: int = 1000
	subjects: tuple[str, ...] = ("Math", "Science", "English", "History", "Geography")


def prompt_for_csv_paths(config: Config) -> Config:
	if not sys.stdin.isatty():
		return config

	marks_path = input(f"Marks CSV path [{config.marks_path}]: ").strip() or config.marks_path
	attendance_path = input(f"Attendance CSV path [{config.attendance_path}]: ").strip() or config.attendance_path
	demographics_path = (
		input(f"Demographics CSV path [{config.demographics_path}]: ").strip() or config.demographics_path
	)

	config.marks_path = marks_path
	config.attendance_path = attendance_path
	config.demographics_path = demographics_path
	return config


def infer_subjects(marks_df: pd.DataFrame, config: Config) -> Config:
	base_columns = {"Student_ID", "Name", "Class"}
	subjects = [col for col in marks_df.columns if col not in base_columns]
	if subjects:
		config.subjects = tuple(subjects)
	return config


def generate_synthetic_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	rng = np.random.default_rng(config.random_seed)
	student_ids = [f"S{str(i + 1).zfill(4)}" for i in range(config.synth_students)]
	first_names = ["John", "Emma", "Michael", "Sarah", "David", "Ava", "Noah", "Olivia"]
	last_names = ["Smith", "Wilson", "Brown", "Johnson", "Lee", "Davis", "Taylor", "Clark"]
	names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in student_ids]
	classes = rng.choice([8, 9, 10], size=config.synth_students)

	base_scores = rng.normal(loc=65.4, scale=16.0, size=config.synth_students)
	marks = {}
	for subject in config.subjects:
		subject_scores = base_scores + rng.normal(loc=0, scale=8.0, size=config.synth_students)
		marks[subject] = np.clip(subject_scores, 0, 100).round(1)

	marks_df = pd.DataFrame(
		{
			"Student_ID": student_ids,
			"Name": names,
			"Class": classes,
			**marks,
		}
	)

	attendance_base = 82.7 + (base_scores - 65.4) * 0.25 + rng.normal(loc=0, scale=8.5, size=config.synth_students)
	attendance_percent = np.clip(attendance_base, 55, 100)
	total_days = rng.integers(180, 201, size=config.synth_students)
	days_present = (total_days * (attendance_percent / 100)).astype(int)

	attendance_df = pd.DataFrame(
		{
			"Student_ID": student_ids,
			"Days_Present": days_present,
			"Total_Days": total_days,
		}
	)

	demographics_df = pd.DataFrame(
		{
			"Student_ID": student_ids,
			"Gender": rng.choice(["Male", "Female"], size=config.synth_students),
			"Parent_Education": rng.choice(
				["High School", "Bachelor", "Master", "PhD"],
				size=config.synth_students,
				p=[0.35, 0.4, 0.2, 0.05],
			),
			"Lunch_Type": rng.choice(["Standard", "Free/Reduced"], size=config.synth_students, p=[0.7, 0.3]),
		}
	)

	return marks_df, attendance_df, demographics_df


def load_or_generate_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	files_exist = (
		os.path.exists(config.marks_path)
		and os.path.exists(config.attendance_path)
		and os.path.exists(config.demographics_path)
	)

	if files_exist:
		marks_df = pd.read_csv(config.marks_path)
		attendance_df = pd.read_csv(config.attendance_path)
		demographics_df = pd.read_csv(config.demographics_path)
		return marks_df, attendance_df, demographics_df

	marks_df, attendance_df, demographics_df = generate_synthetic_data(config)
	marks_df.to_csv(config.marks_path, index=False)
	attendance_df.to_csv(config.attendance_path, index=False)
	demographics_df.to_csv(config.demographics_path, index=False)
	return marks_df, attendance_df, demographics_df


def build_merged_dataset(
	marks_df: pd.DataFrame, attendance_df: pd.DataFrame, demographics_df: pd.DataFrame, config: Config
) -> pd.DataFrame:
	merged = marks_df.merge(attendance_df, on="Student_ID").merge(demographics_df, on="Student_ID")
	merged["Attendance_Percentage"] = (merged["Days_Present"] / merged["Total_Days"] * 100).round(1)
	merged["Average_Mark"] = merged[list(config.subjects)].mean(axis=1).round(1)
	return merged


def summarize_analysis(merged: pd.DataFrame, config: Config) -> dict:
	all_pass = (merged[list(config.subjects)] >= config.pass_threshold).all(axis=1)
	pass_rate = all_pass.mean() * 100
	avg_mark = merged["Average_Mark"].mean()
	std_mark = merged["Average_Mark"].std()
	avg_attendance = merged["Attendance_Percentage"].mean()
	correlation = merged["Attendance_Percentage"].corr(merged["Average_Mark"])

	low_marks = merged["Average_Mark"] < config.pass_threshold
	poor_attendance = merged["Attendance_Percentage"] < config.attendance_threshold
	at_risk_mask = low_marks | poor_attendance
	at_risk_count = int(at_risk_mask.sum())

	return {
		"total_students": len(merged),
		"pass_rate": pass_rate,
		"avg_mark": avg_mark,
		"std_mark": std_mark,
		"avg_attendance": avg_attendance,
		"at_risk_count": at_risk_count,
		"at_risk_percent": at_risk_count / len(merged) * 100,
		"correlation": correlation,
		"at_risk_mask": at_risk_mask,
	}


def build_at_risk_table(merged: pd.DataFrame, config: Config) -> pd.DataFrame:
	low_marks = merged["Average_Mark"] < config.pass_threshold
	poor_attendance = merged["Attendance_Percentage"] < config.attendance_threshold

	def risk_reason(row: pd.Series) -> str:
		reasons = []
		if row["Average_Mark"] < config.pass_threshold:
			reasons.append("Low Marks")
		if row["Attendance_Percentage"] < config.attendance_threshold:
			reasons.append("Poor Attendance")
		return ", ".join(reasons)

	at_risk = merged[low_marks | poor_attendance].copy()
	at_risk["Risk"] = at_risk.apply(risk_reason, axis=1)
	at_risk = at_risk.sort_values(["Average_Mark", "Attendance_Percentage"]).reset_index(drop=True)
	return at_risk


def create_visualizations(merged: pd.DataFrame, config: Config) -> None:
	sns.set_theme(style="whitegrid")

	performance_bins = [0, 40, 60, 80, 100]
	performance_labels = ["Low", "Average", "Good", "Excellent"]
	merged["Performance_Band"] = pd.cut(merged["Average_Mark"], bins=performance_bins, labels=performance_labels)
	performance_counts = merged["Performance_Band"].value_counts().reindex(performance_labels)

	fig = plt.figure(figsize=(16, 10))
	grid = fig.add_gridspec(2, 3)

	ax1 = fig.add_subplot(grid[0, 0])
	performance_counts.plot(kind="bar", ax=ax1, color=["#c0392b", "#f39c12", "#27ae60", "#2980b9"])
	ax1.set_title("Performance Distribution")
	ax1.set_xlabel("Band")
	ax1.set_ylabel("Students")

	ax2 = fig.add_subplot(grid[0, 1])
	sns.scatterplot(
		data=merged,
		x="Attendance_Percentage",
		y="Average_Mark",
		hue="Performance_Band",
		palette="viridis",
		ax=ax2,
		legend=False,
	)
	ax2.set_title("Marks vs Attendance")

	ax3 = fig.add_subplot(grid[0, 2])
	sns.boxplot(data=merged[list(config.subjects)], ax=ax3)
	ax3.set_title("Subject-wise Box Plots")

	ax4 = fig.add_subplot(grid[1, 0])
	gender_avg = merged.groupby("Gender")["Average_Mark"].mean().reset_index()
	sns.barplot(
		data=gender_avg,
		x="Gender",
		y="Average_Mark",
		hue="Gender",
		ax=ax4,
		palette="Set2",
		legend=False,
	)
	ax4.set_title("Average Marks by Gender")

	ax5 = fig.add_subplot(grid[1, 1])
	lunch_avg = merged.groupby("Lunch_Type")["Average_Mark"].mean().reset_index()
	sns.barplot(
		data=lunch_avg,
		x="Lunch_Type",
		y="Average_Mark",
		hue="Lunch_Type",
		ax=ax5,
		palette="Set3",
		legend=False,
	)
	ax5.set_title("Average Marks by Lunch Type")
	ax5.tick_params(axis="x", rotation=15)

	ax6 = fig.add_subplot(grid[1, 2])
	corr = merged[list(config.subjects) + ["Attendance_Percentage", "Average_Mark"]].corr()
	sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax6)
	ax6.set_title("Correlation Heatmap")

	fig.tight_layout()
	fig.savefig(config.dashboard_path, dpi=150)
	plt.close(fig)


def save_text_report(summary: dict, at_risk: pd.DataFrame, config: Config) -> None:
	lines = [
		"STUDENT PERFORMANCE ANALYTICS SUMMARY",
		"====================================",
		f"Total Students: {summary['total_students']}",
		f"Overall Pass Rate: {summary['pass_rate']:.1f}%",
		f"Average Marks: {summary['avg_mark']:.1f} (Std Dev: {summary['std_mark']:.1f})",
		f"Average Attendance: {summary['avg_attendance']:.1f}%",
		f"At-Risk Students: {summary['at_risk_count']} ({summary['at_risk_percent']:.1f}%)",
		f"Correlation (Attendance vs Marks): {summary['correlation']:.3f}",
		"",
		"TOP 10 AT-RISK STUDENTS",
		"------------------------",
	]
	preview = at_risk.head(10)
	for _, row in preview.iterrows():
		lines.append(
			f"{row['Student_ID']} | {row['Name']} | Avg: {row['Average_Mark']:.1f} | "
			f"Attendance: {row['Attendance_Percentage']:.1f}% | Risk: {row['Risk']}"
		)
	with open(config.text_report_path, "w", encoding="utf-8") as handle:
		handle.write("\n".join(lines))


def save_excel_report(merged: pd.DataFrame, summary: dict, at_risk: pd.DataFrame, config: Config) -> bool:
	summary_df = pd.DataFrame(
		{
			"Metric": [
				"Total Students",
				"Overall Pass Rate",
				"Average Marks",
				"Marks Std Dev",
				"Average Attendance",
				"At-Risk Students",
				"At-Risk Percent",
				"Correlation (Attendance vs Marks)",
			],
			"Value": [
				summary["total_students"],
				f"{summary['pass_rate']:.1f}%",
				f"{summary['avg_mark']:.1f}",
				f"{summary['std_mark']:.1f}",
				f"{summary['avg_attendance']:.1f}%",
				summary["at_risk_count"],
				f"{summary['at_risk_percent']:.1f}%",
				f"{summary['correlation']:.3f}",
			],
		}
	)

	try:
		with pd.ExcelWriter(config.excel_report_path) as writer:
			merged.to_excel(writer, sheet_name="merged_data", index=False)
			summary_df.to_excel(writer, sheet_name="summary", index=False)
			at_risk.to_excel(writer, sheet_name="at_risk", index=False)
		return True
	except Exception:
		return False


def save_pdf_report(summary: dict, config: Config) -> None:
	with PdfPages(config.pdf_report_path) as pdf:
		fig = plt.figure(figsize=(8.5, 11))
		fig.suptitle("Student Performance Report", fontsize=16, y=0.98)
		text = (
			f"Total Students: {summary['total_students']}\n"
			f"Overall Pass Rate: {summary['pass_rate']:.1f}%\n"
			f"Average Marks: {summary['avg_mark']:.1f} (Std Dev: {summary['std_mark']:.1f})\n"
			f"Average Attendance: {summary['avg_attendance']:.1f}%\n"
			f"At-Risk Students: {summary['at_risk_count']} ({summary['at_risk_percent']:.1f}%)\n"
			f"Correlation (Attendance vs Marks): {summary['correlation']:.3f}"
		)
		fig.text(0.1, 0.8, text, fontsize=12)
		pdf.savefig(fig)
		plt.close(fig)

		if os.path.exists(config.dashboard_path):
			dashboard = plt.imread(config.dashboard_path)
			fig2, ax = plt.subplots(figsize=(11, 8.5))
			ax.imshow(dashboard)
			ax.axis("off")
			pdf.savefig(fig2)
			plt.close(fig2)


def print_console_output(
	marks_df: pd.DataFrame,
	attendance_df: pd.DataFrame,
	demographics_df: pd.DataFrame,
	merged: pd.DataFrame,
	summary: dict,
	at_risk: pd.DataFrame,
	config: Config,
	excel_ok: bool,
) -> None:
	status_ok = "✅"
	status_warn = "⚠"
	print(f"{status_ok} Marks data loaded: {len(marks_df)} students, {marks_df.shape[1]} columns")
	print(f"{status_ok} Attendance data loaded: {len(attendance_df)} records")
	print(f"{status_ok} Demographics data loaded: {len(demographics_df)} students")
	print(f"{status_ok} Datasets merged successfully. Total records: {len(merged)}")
	columns_preview = [
		"Student_ID",
		"Name",
		"Class",
		"Math",
		"Science",
		"English",
		"Days_Present",
		"Total_Days",
		"Attendance_Percentage",
		"Gender",
		"Parent_Education",
		"Lunch_Type",
	]
	if all(col in merged.columns for col in columns_preview):
		print(f"{status_ok} Columns in merged dataset: {columns_preview}")
	else:
		print(f"{status_ok} Columns in merged dataset: {list(merged.columns)}")
	print()

	print(f"Total Students Analyzed: {summary['total_students']}")
	print(f"Overall Pass Rate: {summary['pass_rate']:.1f}%")
	print(f"Average Marks: {summary['avg_mark']:.1f} (Std Dev: {summary['std_mark']:.1f})")
	print(f"Average Attendance: {summary['avg_attendance']:.1f}%")
	print(f"At-Risk Students: {summary['at_risk_count']} ({summary['at_risk_percent']:.1f}%)")
	print(f"Correlation (Attendance vs Marks): {summary['correlation']:.3f}")
	print()

	print("Top 5 At-Risk Students:")
	preview = at_risk.head(5)
	for _, row in preview.iterrows():
		print(
			f"{row['Student_ID']} | {row['Name']} | Avg: {row['Average_Mark']:.1f} | "
			f"Attendance: {row['Attendance_Percentage']:.1f}% | Risk: {row['Risk']}"
		)
	print()

	print(f"{status_ok} Generated 5 visualizations:")
	print("1. Performance Distribution (pie/bar charts)")
	print("2. Marks vs Attendance Scatter Plot")
	print("3. Subject-wise Box Plots")
	print("4. Demographic Analysis Charts")
	print("5. Correlation Heatmap")
	print(f"{status_ok} Dashboard saved to {config.dashboard_path}")
	print()

	print(f"{status_ok} Generating comprehensive reports...")
	print(f"{status_ok} Text report saved to {config.text_report_path}")
	if excel_ok:
		print(f"{status_ok} Excel report saved to {config.excel_report_path}")
	else:
		print(f"{status_warn} Excel report not generated (missing Excel engine).")
	print(f"{status_ok} PDF report saved to {config.pdf_report_path}")
	print(f"{status_ok} Reports generated successfully!")


def main() -> None:
	config = prompt_for_csv_paths(Config())
	marks_df, attendance_df, demographics_df = load_or_generate_data(config)
	config = infer_subjects(marks_df, config)
	merged = build_merged_dataset(marks_df, attendance_df, demographics_df, config)
	summary = summarize_analysis(merged, config)
	at_risk = build_at_risk_table(merged, config)

	create_visualizations(merged, config)
	save_text_report(summary, at_risk, config)
	excel_ok = save_excel_report(merged, summary, at_risk, config)
	save_pdf_report(summary, config)

	print_console_output(
		marks_df,
		attendance_df,
		demographics_df,
		merged,
		summary,
		at_risk,
		config,
		excel_ok,
	)


if __name__ == "__main__":
	main()
