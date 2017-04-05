<services>
	<service name="tar_scm">
		<param name="scm">git</param>
		<param name="url">https://github.com/CESNET/UltraGrid.git</param>
		<param name="version">1.3</param>
		<param name="revision">nightly</param>
		<param name="filename">ultragrid-git</param>
		<param name="package-meta">yes</param>
		<param name="submodules">enable</param>
	</service>
	<service name="extract_file">
		<param name="archive">*ultragrid*.tar</param>
		<param name="files">*/package_specs/ultragrid-proprietary-drivers/*</param>
	</service>
	<service name="download_url">
		<param name="protocol">http</param>
		<param name="host">localhost</param>
		<param name="path">path-to-drivers-archive.tar</param>
		<param name="filename">drivers.tar</param>
	</service>
	<service name="recompress">
		<param name="file">*drivers.tar</param>
		<param name="compression">gz</param>
	</service>
</services>
