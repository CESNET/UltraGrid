<services>
	<service name="tar_scm">
		<param name="scm">git</param>
		<param name="url">https://github.com/CESNET/UltraGrid.git</param>
		<param name="version">1.4</param>
		<param name="revision">nightly</param>
		<param name="filename">ultragrid-nightly</param>
		<param name="package-meta">yes</param>
		<param name="submodules">enable</param>
	</service>
	<service name="extract_file">
		<param name="archive">*ultragrid*.tar</param>
		<param name="files">*/package_specs/ultragrid-nightly/*</param>
	</service>
	<service name="recompress">
		<param name="file">*ultragrid*.tar</param>
		<param name="compression">bz2</param>
	</service>
<!-- dummy commit c9f8af3beed51420d68d24a9206a18f5f269ca5d -->
</services>
