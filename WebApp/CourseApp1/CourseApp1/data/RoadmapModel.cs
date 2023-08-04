using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(RoadmapMetadata))]
    public partial class Roadmap
    {
    }
    public class RoadmapMetadata
    {
        public int Id { get; set; }
        [Display(Name ="RoadMap")]
        public string Name { get; set; }
    }
}